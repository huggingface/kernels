use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use cuda_core::{launch_kernel, CudaContext, CudaFunction, CudaModule};
use tvm_ffi::error::{Error, ErrorKind, Result, RUNTIME_ERROR, TYPE_ERROR, VALUE_ERROR};
use tvm_ffi::tvm_ffi_sys::dlpack::{DLDataTypeCode, DLDeviceType};
use tvm_ffi::{current_stream, Tensor};

const KERNELS_PTX: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/kernels-ptx/relu_kernels.ptx"
));

const THREADS_PER_BLOCK: u32 = 256;

fn err(kind: ErrorKind<'_>, message: &str) -> Error {
    Error::new(kind, message, "")
}

struct Runtime {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    relu_fwd: CudaFunction,
}

fn runtime(device_id: i32) -> Result<&'static Runtime> {
    static RUNTIMES: OnceLock<Mutex<HashMap<i32, &'static Runtime>>> = OnceLock::new();
    let mut runtimes = RUNTIMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(runtime) = runtimes.get(&device_id) {
        return Ok(runtime);
    }

    let ctx = CudaContext::new(device_id as usize).map_err(|e| {
        err(
            RUNTIME_ERROR,
            &format!("cuda context (device {device_id}): {e:?}"),
        )
    })?;
    let module = ctx
        .load_module_from_ptx_src(KERNELS_PTX)
        .map_err(|e| err(RUNTIME_ERROR, &format!("load kernels PTX: {e:?}")))?;
    let relu_fwd = module
        .load_function("relu_fwd")
        .map_err(|e| err(RUNTIME_ERROR, &format!("load relu_fwd: {e:?}")))?;

    let runtime: &'static Runtime = Box::leak(Box::new(Runtime {
        ctx,
        _module: module,
        relu_fwd,
    }));
    runtimes.insert(device_id, runtime);
    Ok(runtime)
}

fn check(name: &str, tensor: &Tensor) -> Result<()> {
    if tensor.device().device_type != DLDeviceType::kDLCUDA {
        return Err(err(TYPE_ERROR, &format!("{name} must be a CUDA tensor")));
    }
    let dtype = tensor.dtype();
    if dtype.code != DLDataTypeCode::kDLFloat as u8 || dtype.bits != 32 || dtype.lanes != 1 {
        return Err(err(TYPE_ERROR, &format!("{name} must be float32")));
    }
    if !tensor.is_contiguous() {
        return Err(err(VALUE_ERROR, &format!("{name} must be contiguous")));
    }
    Ok(())
}

fn relu(x: Tensor, out: Tensor) -> Result<()> {
    check("x", &x)?;
    check("out", &out)?;

    let device = x.device();
    if out.device().device_id != device.device_id {
        return Err(err(VALUE_ERROR, "x and out must be on the same device"));
    }
    if x.numel() != out.numel() {
        return Err(err(VALUE_ERROR, "x and out must have the same size"));
    }

    let n = x.numel() as u64;
    if n == 0 {
        return Ok(());
    }
    let runtime = runtime(device.device_id)?;

    let mut x_ptr = x.data_ptr() as u64;
    let mut x_len = n;
    let mut out_ptr = out.data_ptr() as u64;
    let mut out_len = n;
    let mut params = [
        (&mut x_ptr as *mut u64).cast::<c_void>(),
        (&mut x_len as *mut u64).cast(),
        (&mut out_ptr as *mut u64).cast(),
        (&mut out_len as *mut u64).cast(),
    ];

    runtime
        .ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind context: {e:?}")))?;

    // SAFETY: `params` matches `relu_fwd`'s two (pointer, length) pairs.
    unsafe {
        launch_kernel(
            runtime.relu_fwd.cu_function(),
            (n.div_ceil(THREADS_PER_BLOCK as u64) as u32, 1, 1),
            (THREADS_PER_BLOCK, 1, 1),
            0,
            current_stream(&device).cast(),
            &mut params,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("launch relu_fwd: {e:?}")))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(relu, relu);
