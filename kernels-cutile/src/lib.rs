//! Runtime support for kernel-builder's `rust-cutile` kernels.
//!
//! cuTile JIT-compiles kernels on their first launch. [`Gpu::new`] binds the
//! device's primary context, which Torch also uses, borrows the caller's
//! current stream and enables cuTile's on-disk cubin cache, so that compiled
//! kernels are reused across processes. [`Gpu::tensor`] wraps tvm-ffi tensors
//! as cuTile tensors without taking ownership, and [`Gpu::launch`] queues a
//! kernel on the borrowed stream.
//!
//! ```ignore
//! fn relu(x: Tensor, out: Tensor) -> tvm_ffi::Result<()> {
//!     let gpu = kernels_cutile::Gpu::new(x.device())?;
//!     unsafe {
//!         let x = gpu.tensor::<f32>(&x)?;
//!         let out = gpu.tensor::<f32>(&out)?;
//!         gpu.launch(kernels::relu(out.partition([1024]), x))?;
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Disk cache
//!
//! Compiled kernels are cached in cuTile's default location
//! (`$XDG_CACHE_HOME/cutile/kernels` or `~/.cache/cutile/kernels`), which
//! cuTile restricts to the current user. Set `KERNELS_CUTILE_DISABLE_CACHE` to
//! `1`, `true`, `yes`, or `on` to disable the disk cache.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ffi::OsStr;
use std::sync::{Arc, Mutex, Once};

use cuda_core::{Context, DType, DTypeId, Stream};
use cutile::jit_cache;
use cutile::prelude::DeviceOp;
use tvm_ffi::{Error, RUNTIME_ERROR, Result, Tensor, VALUE_ERROR};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataTypeCode, DLDevice, DLDeviceType};

const DISABLE_CACHE_ENV: &str = "KERNELS_CUTILE_DISABLE_CACHE";

/// Converts a CUDA error, such as a failed launch, to a tvm-ffi error.
pub fn cuda_error(e: impl std::fmt::Debug) -> Error {
    Error::new(RUNTIME_ERROR, &format!("{e:?}"), "")
}

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}

/// The caller's current stream and the primary context of one device.
///
/// The stream is borrowed for the current call, so don't keep a `Gpu` after
/// returning to the caller.
pub struct Gpu {
    pub ctx: Arc<Context>,
    pub stream: Arc<Stream>,
    device_id: i32,
}

impl Gpu {
    /// Binds the primary context of `device` to the current thread and
    /// borrows the caller's current stream on it.
    pub fn new(device: DLDevice) -> Result<Gpu> {
        if device.device_type != DLDeviceType::kDLCUDA {
            return Err(value_error("expected a CUDA device"));
        }
        setup_jit_cache();

        static CONTEXTS: Mutex<BTreeMap<i32, Arc<Context>>> = Mutex::new(BTreeMap::new());
        let ctx = match CONTEXTS.lock().unwrap().entry(device.device_id) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let ordinal = usize::try_from(device.device_id)
                    .map_err(|_| value_error("invalid CUDA device id"))?;
                // Retains the primary context, which Torch also uses.
                let ctx = Context::new(ordinal).map_err(cuda_error)?;
                entry.insert(ctx).clone()
            }
        };
        ctx.bind_to_thread().map_err(cuda_error)?;

        // Torch's current stream, which Torch keeps owning.
        let stream = unsafe {
            let raw = TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id);
            Stream::borrow_raw(raw, &ctx)
        };

        Ok(Gpu {
            ctx,
            stream,
            device_id: device.device_id,
        })
    }

    /// Borrows a contiguous `T` tensor on this device as a cuTile tensor with
    /// the same shape.
    ///
    /// # Safety
    ///
    /// The tvm-ffi tensor must stay alive and must not be accessed outside of
    /// `self.stream` until the work queued on the returned tensor completes.
    pub unsafe fn tensor<T: DType>(&self, tensor: &Tensor) -> Result<cutile::tensor::Tensor<T>> {
        let device = tensor.device();
        if device.device_type != DLDeviceType::kDLCUDA || device.device_id != self.device_id {
            return Err(value_error(&format!(
                "tensors must be on cuda:{}",
                self.device_id
            )));
        }

        let (code, bits) = dlpack_dtype(T::DTYPE).ok_or_else(|| {
            value_error(&format!(
                "unsupported element type: {}",
                std::any::type_name::<T>()
            ))
        })?;
        let dtype = tensor.dtype();
        if dtype.code != code as u8 || dtype.bits != bits || dtype.lanes != 1 {
            return Err(value_error(&format!(
                "expected a {} tensor",
                std::any::type_name::<T>()
            )));
        }

        if !tensor.is_contiguous() {
            return Err(value_error("tensors must be contiguous"));
        }

        let shape = tensor
            .shape()
            .iter()
            .map(|&dim| i32::try_from(dim))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| value_error("tensor dimensions must be smaller than 2^31"))?;
        let strides = contiguous_strides(&shape)
            .ok_or_else(|| value_error("tensors must have fewer than 2^31 elements"))?;

        // NOTE: tvm-ffi does not expose the DLPack `byte_offset`, the assumption
        //       that it is 0 holds for Torch, but what about other frameworks?
        let ptr = tensor.data_ptr() as u64;
        // SAFETY: the tensor holds a contiguous `T` array of this shape at
        // `ptr` on this device. The caller upholds the liveness and aliasing
        // requirements.
        Ok(unsafe {
            cutile::tensor::Tensor::borrow_raw_parts(ptr, self.ctx.clone(), shape, strides)
        })
    }

    /// Queues `op` on the caller's current stream without waiting for it.
    ///
    /// # Safety
    ///
    /// All tensors that `op` accesses must stay alive and must not be accessed
    /// outside of `self.stream` until the work completes.
    pub unsafe fn launch<O: DeviceOp>(&self, op: O) -> Result<<O as DeviceOp>::Output> {
        unsafe { op.async_on(&self.stream) }.map_err(cuda_error)
    }
}

/// Row-major strides of `shape`, or `None` if they do not fit `i32`.
fn contiguous_strides(shape: &[i32]) -> Option<Vec<i32>> {
    let mut strides = vec![0; shape.len()];
    let mut stride: i32 = 1;
    for (dim, out) in shape.iter().zip(strides.iter_mut()).rev() {
        *out = stride;
        stride = stride.checked_mul((*dim).max(1))?;
    }
    Some(strides)
}

/// The DLPack type code and bit width of a cuTile element type.
fn dlpack_dtype(dtype: DTypeId) -> Option<(DLDataTypeCode, u8)> {
    use DLDataTypeCode::*;
    Some(match dtype {
        DTypeId::Bool => (kDLBool, 8),
        DTypeId::U8 => (kDLUInt, 8),
        DTypeId::U16 => (kDLUInt, 16),
        DTypeId::U32 => (kDLUInt, 32),
        DTypeId::U64 => (kDLUInt, 64),
        DTypeId::I8 => (kDLInt, 8),
        DTypeId::I16 => (kDLInt, 16),
        DTypeId::I32 => (kDLInt, 32),
        DTypeId::I64 => (kDLInt, 64),
        DTypeId::F16 => (kDLFloat, 16),
        DTypeId::BF16 => (kDLBfloat, 16),
        DTypeId::F32 => (kDLFloat, 32),
        DTypeId::F64 => (kDLFloat, 64),
        DTypeId::F8E4M3FN => (kDLFloat8_e4m3fn, 8),
        DTypeId::F8E5M2 => (kDLFloat8_e5m2, 8),
        DTypeId::F8E8M0FNU => (kDLFloat8_e8m0fnu, 8),
        // No DLPack equivalent (TF32, F8E5M3FNU) or not a direct match
        // (F4E2M1FNX2 packs two values per byte).
        DTypeId::TF32 | DTypeId::F8E5M3FNU | DTypeId::F4E2M1FNX2 => return None,
    })
}

/// Enables the on-disk cubin cache once per process. Failures only disable
/// the cache, they never fail a launch.
fn setup_jit_cache() {
    static SETUP: Once = Once::new();
    SETUP.call_once(|| {
        if let Err(e) = try_setup_jit_cache() {
            eprintln!("kernels-cutile: JIT disk cache disabled: {e}");
        }
    });
}

fn try_setup_jit_cache() -> std::io::Result<()> {
    if std::env::var_os(DISABLE_CACHE_ENV).is_some_and(|value| is_true(&value)) {
        return Ok(());
    }
    jit_cache::enable_default()
}

fn is_true(value: &OsStr) -> bool {
    value.to_str().is_some_and(|value| {
        ["1", "true", "yes", "on"]
            .iter()
            .any(|t| value.eq_ignore_ascii_case(t))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dlpack_dtypes() {
        assert_eq!(
            dlpack_dtype(DTypeId::F32),
            Some((DLDataTypeCode::kDLFloat, 32))
        );
        assert_eq!(
            dlpack_dtype(DTypeId::BF16),
            Some((DLDataTypeCode::kDLBfloat, 16))
        );
        assert_eq!(
            dlpack_dtype(DTypeId::Bool),
            Some((DLDataTypeCode::kDLBool, 8))
        );
        assert_eq!(
            dlpack_dtype(DTypeId::U16),
            Some((DLDataTypeCode::kDLUInt, 16))
        );
        assert_eq!(
            dlpack_dtype(DTypeId::F8E4M3FN),
            Some((DLDataTypeCode::kDLFloat8_e4m3fn, 8))
        );
        assert_eq!(dlpack_dtype(DTypeId::TF32), None);
        assert_eq!(dlpack_dtype(DTypeId::F4E2M1FNX2), None);
    }

    #[test]
    fn dlpack_dtypes_match_element_sizes() {
        fn check<T: DType>() {
            let (_, bits) = dlpack_dtype(T::DTYPE).unwrap();
            assert_eq!(bits as usize, std::mem::size_of::<T>() * 8);
        }
        check::<bool>();
        check::<u8>();
        check::<i32>();
        check::<i64>();
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), Some(vec![12, 4, 1]));
        assert_eq!(contiguous_strides(&[]), Some(vec![]));
        assert_eq!(contiguous_strides(&[3, 0, 2]), Some(vec![2, 2, 1]));
        assert_eq!(contiguous_strides(&[1 << 16, 1 << 16, 1]), None);
    }

    #[test]
    fn true_values() {
        for value in ["1", "true", "TRUE", "Yes", "on"] {
            assert!(is_true(OsStr::new(value)), "{value}");
        }
        for value in ["", "0", "false", "off", "no", "2"] {
            assert!(!is_true(OsStr::new(value)), "{value}");
        }
    }
}
