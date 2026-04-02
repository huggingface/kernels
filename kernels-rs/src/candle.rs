use std::ffi::c_void;

use candle_core::{CpuStorage, DType, Device, Storage, Tensor};

use crate::KernelModule;
use crate::backend::BackendKind;
use crate::error::{Error, Result};
use crate::tvm_ffi::{self, DLDataType, DLTensor, TVMFFIAny};

fn err(msg: impl Into<String>) -> Error {
    Error::Kernel(msg.into())
}

impl BackendKind {
    pub fn to_candle_supported(self) -> Self {
        match self {
            #[cfg(feature = "candle-cuda")]
            BackendKind::Cuda => BackendKind::Cuda,
            #[cfg(not(feature = "candle-cuda"))]
            BackendKind::Cuda => BackendKind::Cpu,
            other => other,
        }
    }
}

impl TryFrom<BackendKind> for Device {
    type Error = Error;

    fn try_from(kind: BackendKind) -> Result<Self> {
        match kind {
            BackendKind::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "candle-cuda")]
            BackendKind::Cuda => Device::new_cuda(0).map_err(Into::into),
            #[cfg(not(feature = "candle-cuda"))]
            BackendKind::Cuda => Ok(Device::Cpu),
            BackendKind::Xpu => Ok(Device::Cpu),
        }
    }
}

impl TryFrom<&Device> for BackendKind {
    type Error = Error;

    fn try_from(device: &Device) -> Result<Self> {
        match device {
            Device::Cpu => Ok(BackendKind::Cpu),
            #[cfg(feature = "candle-cuda")]
            Device::Cuda(_) => Ok(BackendKind::Cuda),
            #[cfg(not(feature = "candle-cuda"))]
            Device::Cuda(_) => Err(err(
                "CUDA candle device is not supported without the `candle-cuda` feature",
            )),
            Device::Metal(_) => Err(err("Metal candle devices are not supported")),
        }
    }
}

struct PreparedArg {
    tensor: DLTensor,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

enum PreparedCallArg {
    Tensor(PreparedArg),
    Scalar(TVMFFIAny),
}

impl PreparedCallArg {
    fn as_tvm_arg(&mut self) -> TVMFFIAny {
        match self {
            Self::Tensor(arg) => TVMFFIAny::from_dltensor(&mut arg.tensor as *mut DLTensor),
            Self::Scalar(arg) => *arg,
        }
    }
}

#[derive(Clone, Copy)]
pub enum KernelArg<'a> {
    Tensor(&'a Tensor),
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl<'a> From<&'a Tensor> for KernelArg<'a> {
    fn from(value: &'a Tensor) -> Self {
        Self::Tensor(value)
    }
}

impl From<bool> for KernelArg<'_> {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i32> for KernelArg<'_> {
    fn from(value: i32) -> Self {
        Self::Int(value.into())
    }
}

impl From<i64> for KernelArg<'_> {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<u32> for KernelArg<'_> {
    fn from(value: u32) -> Self {
        Self::Int(value.into())
    }
}

impl From<f32> for KernelArg<'_> {
    fn from(value: f32) -> Self {
        Self::Float(value.into())
    }
}

impl From<f64> for KernelArg<'_> {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

// Helper to convert a variable number of heterogeneous arguments
// into a slice of `KernelArg`.
//
// Example:
//
// without:
// call("my_kernel", [
//     KernelArg::Tensor(&my_tensor),
//     KernelArg::Int(42),
//     KernelArg::Float(3.14),
// ])
//
// with:
// call("my_kernel", kargs![&my_tensor, 42, 3.14])
//
#[macro_export]
macro_rules! kargs {
    ($($arg:expr),* $(,)?) => {
        [$($crate::candle::KernelArg::from($arg)),*]
    };
}

fn call_backend_kind(args: &[KernelArg<'_>], fallback: BackendKind) -> Result<BackendKind> {
    let mut kind = None;

    for arg in args {
        let KernelArg::Tensor(tensor) = arg else {
            continue;
        };

        let arg_kind = BackendKind::try_from(tensor.device())?;
        match kind {
            Some(prev) if prev != arg_kind => {
                return Err(err("all tensor arguments must use the same backend"));
            }
            Some(_) => {}
            None => kind = Some(arg_kind),
        }
    }

    Ok(kind.unwrap_or(fallback))
}

impl TryFrom<DType> for DLDataType {
    type Error = Error;

    fn try_from(dtype: DType) -> Result<Self> {
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
        Ok(Self {
            code,
            bits,
            lanes: 1,
        })
    }
}

fn cpu_slice_data_ptr<T>(slice: &[T], offset: usize) -> Result<*mut c_void> {
    let slice = slice
        .get(offset..)
        .ok_or_else(|| err("CPU storage offset out of bounds"))?;
    Ok(slice.as_ptr() as *mut c_void)
}

fn cpu_storage_data_ptr(cpu: &CpuStorage, offset: usize) -> Result<*mut c_void> {
    match cpu {
        CpuStorage::U8(v) => cpu_slice_data_ptr(v, offset),
        CpuStorage::U32(v) => cpu_slice_data_ptr(v, offset),
        CpuStorage::I64(v) => cpu_slice_data_ptr(v, offset),
        CpuStorage::BF16(v) => cpu_slice_data_ptr(v, offset),
        CpuStorage::F16(v) => cpu_slice_data_ptr(v, offset),
        CpuStorage::F32(v) => cpu_slice_data_ptr(v, offset),
        CpuStorage::F64(v) => cpu_slice_data_ptr(v, offset),
        _ => Err(err("unsupported CpuStorage variant")),
    }
}

#[cfg(feature = "candle-cuda")]
fn cuda_slice_data_ptr<T>(
    slice: &cudarc::driver::CudaSlice<T>,
    stream: &cudarc::driver::CudaStream,
    offset: usize,
) -> Result<*mut c_void> {
    use cudarc::driver::DevicePtr;

    // SyncOnDrop records a stream event; the pointer stays valid as long
    // as the caller holds the storage read-guard.
    let view = slice
        .try_slice(offset..)
        .ok_or_else(|| err("CUDA storage offset out of bounds"))?;
    let (device_ptr, _sync) = view.device_ptr(stream);
    Ok(device_ptr as *mut c_void)
}

#[cfg(feature = "candle-cuda")]
fn cuda_storage_data_ptr(cuda: &candle_core::CudaStorage, offset: usize) -> Result<*mut c_void> {
    use candle_core::cuda_backend::CudaStorageSlice as S;

    let stream = cuda.device.cuda_stream();

    match &cuda.slice {
        S::U8(s) => cuda_slice_data_ptr(s, &stream, offset),
        S::U32(s) => cuda_slice_data_ptr(s, &stream, offset),
        S::I64(s) => cuda_slice_data_ptr(s, &stream, offset),
        S::BF16(s) => cuda_slice_data_ptr(s, &stream, offset),
        S::F16(s) => cuda_slice_data_ptr(s, &stream, offset),
        S::F32(s) => cuda_slice_data_ptr(s, &stream, offset),
        S::F64(s) => cuda_slice_data_ptr(s, &stream, offset),
        _ => Err(err("unsupported CudaStorage variant")),
    }
}

fn storage_data_ptr(storage: &Storage, offset: usize) -> Result<*mut c_void> {
    match storage {
        Storage::Cpu(cpu) => cpu_storage_data_ptr(cpu, offset),
        #[cfg(feature = "candle-cuda")]
        Storage::Cuda(cuda) => cuda_storage_data_ptr(cuda, offset),
        #[cfg(not(feature = "candle-cuda"))]
        Storage::Cuda(_) => Err(err(
            "CUDA storage is not supported without the `candle-cuda` feature",
        )),
        Storage::Metal(_) => Err(err("Metal storage is not supported")),
    }
}

pub fn get_kernel(repo_id: &str, version: u32) -> Result<KernelModule> {
    let kind = crate::backend::detect().to_candle_supported();
    crate::get_kernel_for_backend(repo_id, version, kind)
}

pub fn get_local_kernel(repo_path: &std::path::Path) -> Result<KernelModule> {
    let kind = crate::backend::detect().to_candle_supported();
    crate::get_local_kernel_for_backend(repo_path, kind)
}

impl KernelModule {
    pub fn device(&self) -> Result<Device> {
        Device::try_from(self.backend().kind())
    }

    // Tensor arguments are passed as DLTensor views directly into candle
    // storage. Scalar arguments are marshalled as TVM FFI Any POD values.
    pub fn call<'a, A>(&self, func_name: &str, args: A) -> Result<()>
    where
        A: AsRef<[KernelArg<'a>]>,
    {
        let args = args.as_ref();
        let kind = call_backend_kind(args, self.backend().kind())?;

        let symbol = format!("__tvm_ffi_{}_{}", func_name, kind.as_str());
        let func = unsafe { self.get_func(symbol.as_bytes()) }?;

        let contiguous: Vec<Tensor> = args
            .iter()
            .filter_map(|arg| match arg {
                KernelArg::Tensor(tensor) => Some(tensor.contiguous().map_err(Into::into)),
                KernelArg::Bool(_) | KernelArg::Int(_) | KernelArg::Float(_) => None,
            })
            .collect::<Result<_>>()?;

        let guards: Vec<_> = contiguous.iter().map(|t| t.storage_and_layout()).collect();

        let mut tensor_index = 0;
        let mut prepared: Vec<PreparedCallArg> = args
            .iter()
            .map(|arg| match arg {
                KernelArg::Tensor(_) => {
                    let (storage, layout) = &guards[tensor_index];
                    let tensor = &contiguous[tensor_index];
                    tensor_index += 1;

                    let mut shape: Vec<_> = layout.dims().iter().map(|&d| d as i64).collect();
                    let mut strides: Vec<_> = layout.stride().iter().map(|&s| s as i64).collect();
                    let tensor = DLTensor {
                        data: storage_data_ptr(storage, layout.start_offset())?,
                        device: kind.into(),
                        ndim: shape.len() as i32,
                        dtype: tensor.dtype().try_into()?,
                        shape: shape.as_mut_ptr(),
                        strides: strides.as_mut_ptr(),
                        byte_offset: 0,
                    };

                    Ok(PreparedCallArg::Tensor(PreparedArg {
                        tensor,
                        _shape: shape,
                        _strides: strides,
                    }))
                }
                KernelArg::Bool(value) => Ok(PreparedCallArg::Scalar(TVMFFIAny::from_bool(*value))),
                KernelArg::Int(value) => Ok(PreparedCallArg::Scalar(TVMFFIAny::from_int(*value))),
                KernelArg::Float(value) => {
                    Ok(PreparedCallArg::Scalar(TVMFFIAny::from_float(*value)))
                }
            })
            .collect::<Result<_>>()?;

        let tvm_args: Vec<TVMFFIAny> = prepared
            .iter_mut()
            .map(PreparedCallArg::as_tvm_arg)
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
