//! Runtime support for kernel-builder's `rust-cuda` kernels.
//!
//! kernel-builder compiles a kernel crate's PTX for every CUDA capability in
//! `build.toml` and links the fatbin into the extension. [`load`] loads that
//! fatbin into the device's primary context, which Torch also uses, binds it
//! to a `#[cuda_module]` and borrows the caller's current stream.
//! [`Gpu::view`] wraps Torch tensors as `DeviceBuffer`s without taking
//! ownership.
//!
//! ```ignore
//! fn relu(x: Tensor, out: Tensor) -> tvm_ffi::Result<()> {
//!     let gpu = kernels_cuda::load(x.device(), kernels::from_module)?;
//!     let (x, mut out) = (gpu.view::<f32>(&x)?, gpu.view::<f32>(&out)?);
//!     let config = LaunchConfig::for_num_elems(x.len() as u32);
//!     unsafe { gpu.module.relu(&gpu.stream, config, &x, &mut out) }
//!         .map_err(kernels_cuda::cuda_error)
//! }
//! ```

use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, DriverError};
use tvm_ffi::{Error, RUNTIME_ERROR, Result, Tensor, VALUE_ERROR};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataTypeCode, DLDevice, DLDeviceType};

// kernel-builder links the fatbin between these symbols.
unsafe extern "C" {
    static KERNEL_BUILDER_CUDA_FATBIN: u8;
    static KERNEL_BUILDER_CUDA_FATBIN_END: u8;
}

fn fatbin() -> &'static [u8] {
    let start = &raw const KERNEL_BUILDER_CUDA_FATBIN;
    let end = &raw const KERNEL_BUILDER_CUDA_FATBIN_END;
    unsafe { std::slice::from_raw_parts(start, end.offset_from(start) as usize) }
}

/// Converts a CUDA error, such as a failed launch, to a tvm-ffi error.
pub fn cuda_error(e: impl std::fmt::Debug) -> Error {
    Error::new(RUNTIME_ERROR, &format!("{e:?}"), "")
}

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}

/// A loaded module and the caller's current stream on one device.
///
/// The stream is borrowed for the current call, so don't keep a `Gpu` after
/// returning to the caller.
pub struct Gpu<M> {
    pub module: Arc<M>,
    pub stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    device_id: i32,
}

type Loaded = (Arc<CudaContext>, Arc<dyn Any + Send + Sync>);

/// Loads the kernels for `device` once and binds them with `from_module`,
/// the function `#[cuda_module]` generates.
pub fn load<M: Send + Sync + 'static>(
    device: DLDevice,
    from_module: fn(Arc<CudaModule>) -> std::result::Result<M, DriverError>,
) -> Result<Gpu<M>> {
    if device.device_type != DLDeviceType::kDLCUDA {
        return Err(value_error("expected a CUDA device"));
    }
    static LOADED: Mutex<BTreeMap<(i32, TypeId), Loaded>> = Mutex::new(BTreeMap::new());
    let (ctx, module) = match LOADED
        .lock()
        .unwrap()
        .entry((device.device_id, TypeId::of::<M>()))
    {
        Entry::Occupied(entry) => entry.get().clone(),
        Entry::Vacant(entry) => {
            let ctx = CudaContext::new(device.device_id as usize).map_err(cuda_error)?;
            let module = ctx
                .load_module_from_image(fatbin())
                .and_then(from_module)
                .map_err(cuda_error)?;
            entry.insert((ctx, Arc::new(module))).clone()
        }
    };
    let module = module.downcast::<M>().expect("modules are keyed by type");

    ctx.bind_to_thread().map_err(cuda_error)?;
    // Torch's current stream, which Torch keeps owning.
    let stream = unsafe {
        let raw = TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id);
        CudaStream::borrow_raw(raw, &ctx)
    };
    Ok(Gpu {
        module,
        stream,
        ctx,
        device_id: device.device_id,
    })
}

impl<M> Gpu<M> {
    /// Views a contiguous `T` tensor on this device as a `DeviceBuffer<T>`.
    pub fn view<T: Element>(&self, tensor: &Tensor) -> Result<View<T>> {
        let device = tensor.device();
        if device.device_type != DLDeviceType::kDLCUDA || device.device_id != self.device_id {
            return Err(value_error(&format!(
                "tensors must be on cuda:{}",
                self.device_id
            )));
        }
        let dtype = tensor.dtype();
        if dtype.code != T::CODE as u8 || dtype.bits != T::BITS || dtype.lanes != 1 {
            return Err(value_error(&format!(
                "expected a {} tensor",
                std::any::type_name::<T>()
            )));
        }
        if !tensor.is_contiguous() {
            return Err(value_error("tensors must be contiguous"));
        }
        let ptr = tensor.data_ptr() as u64;
        // SAFETY: the tensor owns `numel` elements of `T` at `ptr` on this
        // device. `View` never frees them.
        let buffer = unsafe { DeviceBuffer::from_raw_parts(ptr, tensor.numel(), self.ctx.clone()) };
        Ok(View(ManuallyDrop::new(buffer)))
    }
}

/// A tensor's memory as a `DeviceBuffer`. Dropping it leaves the memory to
/// the tensor.
pub struct View<T>(ManuallyDrop<DeviceBuffer<T>>);

impl<T> Deref for View<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.0
    }
}

impl<T> DerefMut for View<T> {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.0
    }
}

impl<T> Drop for View<T> {
    fn drop(&mut self) {
        let buffer = unsafe { ManuallyDrop::take(&mut self.0) };
        let _ = buffer.into_raw_parts();
    }
}

/// An element type with a DLPack equivalent.
pub trait Element {
    const CODE: DLDataTypeCode;
    const BITS: u8;
}

macro_rules! element {
    ($($ty:ty => $code:ident, $bits:literal;)*) => {
        $(impl Element for $ty {
            const CODE: DLDataTypeCode = DLDataTypeCode::$code;
            const BITS: u8 = $bits;
        })*
    };
}

element! {
    i8 => kDLInt, 8;
    i16 => kDLInt, 16;
    i32 => kDLInt, 32;
    i64 => kDLInt, 64;
    u8 => kDLUInt, 8;
    u16 => kDLUInt, 16;
    u32 => kDLUInt, 32;
    u64 => kDLUInt, 64;
    half::f16 => kDLFloat, 16;
    f32 => kDLFloat, 32;
    f64 => kDLFloat, 64;
    half::bf16 => kDLBfloat, 16;
}
