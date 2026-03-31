use std::ffi::c_void;

use candle_core::{CpuStorage, DType, Device, Storage, Tensor};

use crate::KernelModule;
use crate::backend::BackendKind;
use crate::error::{Error, Result};
use crate::tvm_ffi::{self, DLDataType, DLDevice, DLTensor, TVMFFIAny};

fn err(msg: impl Into<String>) -> Error {
    Error::Kernel(msg.into())
}

impl BackendKind {
    pub fn candle_device(self) -> Result<Device> {
        match self {
            BackendKind::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "candle-cuda")]
            BackendKind::Cuda => Device::new_cuda(0).map_err(Into::into),
            #[cfg(not(feature = "candle-cuda"))]
            BackendKind::Cuda => Ok(Device::Cpu),
            BackendKind::Xpu => Ok(Device::Cpu),
        }
    }

    pub fn candle_supported(self) -> Self {
        match self {
            #[cfg(feature = "candle-cuda")]
            BackendKind::Cuda => BackendKind::Cuda,
            #[cfg(not(feature = "candle-cuda"))]
            BackendKind::Cuda => BackendKind::Cpu,
            other => other,
        }
    }
}

impl From<&Device> for BackendKind {
    fn from(device: &Device) -> Self {
        match device {
            Device::Cpu => BackendKind::Cpu,
            #[cfg(feature = "candle-cuda")]
            Device::Cuda(_) => BackendKind::Cuda,
            #[allow(unreachable_patterns)]
            _ => BackendKind::Cpu,
        }
    }
}

struct PreparedArg {
    data: *mut c_void,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype: DLDataType,
}

fn dtype_to_dl(dtype: DType) -> Result<DLDataType> {
    let (code, bits) = match dtype {
        DType::U8 => (tvm_ffi::DL_UINT, 8),
        DType::U32 => (tvm_ffi::DL_UINT, 32),
        DType::I64 => (tvm_ffi::DL_INT, 64),
        DType::BF16 => (tvm_ffi::DL_BFLOAT, 16),
        DType::F16 => (tvm_ffi::DL_FLOAT, 16),
        DType::F32 => (tvm_ffi::DL_FLOAT, 32),
        DType::F64 => (tvm_ffi::DL_FLOAT, 64),
        other => return Err(err(format!("unsupported dtype: {other:?}"))),
    };
    Ok(DLDataType {
        code,
        bits,
        lanes: 1,
    })
}

fn cpu_storage_data_ptr(cpu: &CpuStorage, offset: usize) -> Result<*mut c_void> {
    macro_rules! ptr {
        ($v:expr) => {
            Ok(unsafe { $v.as_ptr().add(offset) as *mut c_void })
        };
    }
    match cpu {
        CpuStorage::U8(v) => ptr!(v),
        CpuStorage::U32(v) => ptr!(v),
        CpuStorage::I64(v) => ptr!(v),
        CpuStorage::BF16(v) => ptr!(v),
        CpuStorage::F16(v) => ptr!(v),
        CpuStorage::F32(v) => ptr!(v),
        CpuStorage::F64(v) => ptr!(v),
        _ => Err(err("unsupported CpuStorage variant")),
    }
}

#[cfg(feature = "candle-cuda")]
fn cuda_storage_data_ptr(cuda: &candle_core::CudaStorage, offset: usize) -> Result<*mut c_void> {
    use candle_core::cuda_backend::CudaStorageSlice as S;
    use cudarc::driver::DevicePtr;

    let stream = cuda.device.cuda_stream();

    // SyncOnDrop records a stream event; the pointer stays valid as long
    // as the caller holds the storage read-guard.
    macro_rules! ptr {
        ($slice:expr) => {{
            let view = $slice.slice(offset..);
            let (device_ptr, _sync) = view.device_ptr(&stream);
            Ok(device_ptr as *mut c_void)
        }};
    }

    match &cuda.slice {
        S::U8(s) => ptr!(s),
        S::U32(s) => ptr!(s),
        S::I64(s) => ptr!(s),
        S::BF16(s) => ptr!(s),
        S::F16(s) => ptr!(s),
        S::F32(s) => ptr!(s),
        S::F64(s) => ptr!(s),
        _ => Err(err("unsupported CudaStorage variant")),
    }
}

fn extract_data_ptr(storage: &Storage, offset: usize) -> Result<*mut c_void> {
    match storage {
        Storage::Cpu(cpu) => cpu_storage_data_ptr(cpu, offset),
        #[cfg(feature = "candle-cuda")]
        Storage::Cuda(cuda) => cuda_storage_data_ptr(cuda, offset),
        #[allow(unreachable_patterns)]
        _ => Err(err("unsupported storage backend")),
    }
}

pub fn get_kernel(repo_id: &str, version: u32) -> Result<KernelModule> {
    let kind = crate::backend::detect().candle_supported();
    crate::get_kernel_for_backend(repo_id, version, kind)
}

pub fn get_local_kernel(repo_path: &std::path::Path) -> Result<KernelModule> {
    let kind = crate::backend::detect().candle_supported();
    crate::get_local_kernel_for_backend(repo_path, kind)
}

impl KernelModule {
    pub fn device(&self) -> Result<Device> {
        self.backend().kind().candle_device()
    }
}

// Tensors are passed to the kernel as DLPack pointers directly into
// candle's storage - no copies for contiguous tensors.
pub trait CallKernel {
    fn call(&self, func_name: &str, args: &[&Tensor]) -> Result<()>;
}

impl CallKernel for KernelModule {
    fn call(&self, func_name: &str, args: &[&Tensor]) -> Result<()> {
        let kind = args
            .first()
            .map(|t| BackendKind::from(t.device()))
            .unwrap_or(self.backend().kind());

        let symbol = format!("__tvm_ffi_{}_{}", func_name, kind.as_str());
        let func = unsafe { self.get_func(symbol.as_bytes()) }?;

        let contiguous: Vec<Tensor> = args
            .iter()
            .map(|t| t.contiguous().map_err(Into::into))
            .collect::<Result<_>>()?;

        let guards: Vec<_> = contiguous.iter().map(|t| t.storage_and_layout()).collect();

        let mut prepared: Vec<PreparedArg> = guards
            .iter()
            .enumerate()
            .map(|(i, (storage, layout))| {
                Ok(PreparedArg {
                    data: extract_data_ptr(&storage, layout.start_offset())?,
                    shape: layout.dims().iter().map(|&d| d as i64).collect(),
                    strides: layout.stride().iter().map(|&s| s as i64).collect(),
                    dtype: dtype_to_dl(contiguous[i].dtype())?,
                })
            })
            .collect::<Result<_>>()?;

        let device_type = match kind {
            BackendKind::Cpu => tvm_ffi::DL_CPU,
            BackendKind::Cuda => tvm_ffi::DL_CUDA,
            BackendKind::Xpu => tvm_ffi::DL_ONEAPI,
        };

        let mut dl_tensors: Vec<DLTensor> = prepared
            .iter_mut()
            .map(|p| DLTensor {
                data: p.data,
                device: DLDevice {
                    device_type,
                    device_id: 0,
                },
                ndim: p.shape.len() as i32,
                dtype: p.dtype,
                shape: p.shape.as_mut_ptr(),
                strides: p.strides.as_mut_ptr(),
                byte_offset: 0,
            })
            .collect();

        let tvm_args: Vec<TVMFFIAny> = dl_tensors
            .iter_mut()
            .map(|dl| TVMFFIAny::from_dltensor(dl as *mut DLTensor))
            .collect();
        let mut result = TVMFFIAny::none();

        let ret = unsafe {
            func(
                std::ptr::null_mut(),
                tvm_args.as_ptr(),
                tvm_args.len() as i32,
                &mut result,
            )
        };
        if ret != 0 {
            return Err(err(format!("TVM FFI call `{symbol}` failed (rc {ret})")));
        }
        Ok(())
    }
}
