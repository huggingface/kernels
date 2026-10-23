use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{result, sys, CudaContext};
use tvm_ffi::{Error, Result, Tensor, RUNTIME_ERROR, VALUE_ERROR};
use tvm_ffi_sys::dlpack::{DLDataTypeCode, DLDeviceType};
use tvm_ffi_sys::TVMFFIEnvGetStream;

// PTX of the `relu-device` crate, NUL-terminated for `cuModuleLoadData`.
const PTX: &str = concat!(
    include_str!(concat!(env!("KERNEL_BUILDER_PTX_DIR"), "/relu_device.ptx")),
    "\0"
);

struct Relu {
    ctx: Arc<CudaContext>,
    function: sys::CUfunction,
}

unsafe impl Send for Relu {}

fn cuda_error(e: result::DriverError) -> Error {
    Error::new(RUNTIME_ERROR, &format!("{e:?}"), "")
}

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}

// Loads the module once per device into the primary context Torch also uses.
fn with_relu<T>(device: i32, f: impl FnOnce(&Relu) -> Result<T>) -> Result<T> {
    static RELU: OnceLock<Mutex<HashMap<i32, Relu>>> = OnceLock::new();
    let mut loaded = RELU.get_or_init(Default::default).lock().unwrap();
    if !loaded.contains_key(&device) {
        let ctx = CudaContext::new(device as usize).map_err(cuda_error)?;
        ctx.bind_to_thread().map_err(cuda_error)?;
        let function = unsafe {
            let module = result::module::load_data(PTX.as_ptr().cast()).map_err(cuda_error)?;
            result::module::get_function(module, CString::new("relu").unwrap())
                .map_err(cuda_error)?
        };
        loaded.insert(device, Relu { ctx, function });
    }
    f(&loaded[&device])
}

fn relu_rust_cuda(x: Tensor, out: Tensor) -> Result<()> {
    let f32_type = |t: &Tensor| {
        let dtype = t.dtype();
        dtype.code == DLDataTypeCode::kDLFloat as u8 && dtype.bits == 32 && dtype.lanes == 1
    };
    if !f32_type(&x) || !f32_type(&out) {
        return Err(value_error("input and output must be float32"));
    }
    if !x.is_contiguous() || !out.is_contiguous() {
        return Err(value_error("input and output must be contiguous"));
    }
    if x.numel() != out.numel() {
        return Err(value_error(&format!(
            "input and output must have the same number of elements, got {} and {}",
            x.numel(),
            out.numel()
        )));
    }
    let device = x.device();
    if device.device_type != DLDeviceType::kDLCUDA || out.device() != device {
        return Err(value_error(
            "input and output must be on the same CUDA device",
        ));
    }
    if x.numel() == 0 {
        return Ok(());
    }

    with_relu(device.device_id, |relu| {
        relu.ctx.bind_to_thread().map_err(cuda_error)?;
        let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
        // Slices are passed to cuda-oxide kernels as (pointer, length) pairs.
        let (mut x_ptr, mut out_ptr) = (x.data_ptr() as u64, out.data_ptr() as u64);
        let mut len = x.numel() as u64;
        let mut params = [
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut len as *mut u64).cast(),
            (&mut out_ptr as *mut u64).cast(),
            (&mut len as *mut u64).cast(),
        ];
        let blocks = len.div_ceil(256) as u32;
        unsafe {
            result::launch_kernel(
                relu.function,
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(cuda_error)
    })
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(relu_rust_cuda, relu_rust_cuda);
