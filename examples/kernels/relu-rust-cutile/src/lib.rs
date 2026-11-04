use std::sync::Arc;

use cuda_core::{Context, Stream};
use cutile::prelude::{DeviceOp, IntoPartition};
use tvm_ffi::{Error, Result, Tensor, RUNTIME_ERROR, VALUE_ERROR};
use tvm_ffi_sys::dlpack::{DLDataTypeCode, DLDeviceType};
use tvm_ffi_sys::TVMFFIEnvGetStream;

const TILE: usize = 1024;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn relu<const S: [i32; 1]>(out: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tx = x.load_like(out);
        out.store(max_tile(tx, 0.0f32.broadcast(out.shape())));
    }
}

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}

fn cuda_error(e: impl std::fmt::Debug) -> Error {
    Error::new(RUNTIME_ERROR, &format!("{e:?}"), "")
}

fn check_tensor(tensor: &Tensor, device_id: i32) -> Result<()> {
    let device = tensor.device();
    if device.device_type != DLDeviceType::kDLCUDA || device.device_id != device_id {
        return Err(value_error(&format!("tensors must be on cuda:{device_id}")));
    }
    let dtype = tensor.dtype();
    if dtype.code != DLDataTypeCode::kDLFloat as u8 || dtype.bits != 32 || dtype.lanes != 1 {
        return Err(value_error("expected a float32 tensor"));
    }
    if !tensor.is_contiguous() {
        return Err(value_error("tensors must be contiguous"));
    }
    Ok(())
}

/// Borrows a contiguous `f32` tvm-ffi tensor as a flat cuTile tensor.
///
/// # Safety
///
/// The tvm-ffi tensor must stay alive until all work queued on the tensor
/// has completed.
unsafe fn borrow_tensor(
    tensor: &Tensor,
    ctx: &Arc<Context>,
) -> Result<cutile::tensor::Tensor<f32>> {
    let numel = i32::try_from(tensor.numel())
        .map_err(|_| value_error("tensors must have fewer than 2^31 elements"))?;
    Ok(unsafe {
        cutile::tensor::Tensor::borrow_raw_parts(
            tensor.data_ptr() as u64,
            ctx.clone(),
            vec![numel],
            vec![1],
        )
    })
}

fn relu_rust_cutile(x: Tensor, out: Tensor) -> Result<()> {
    if x.numel() != out.numel() {
        return Err(value_error(&format!(
            "input and output must have the same number of elements, got {} and {}",
            x.numel(),
            out.numel()
        )));
    }
    if x.numel() == 0 {
        return Ok(());
    }

    let device_id = x.device().device_id;
    check_tensor(&x, device_id)?;
    check_tensor(&out, device_id)?;

    // TODO: cache the context per device instead of retaining it on every call.
    // `Context::new` retains the primary context, which Torch also uses.
    let ctx = Context::new(device_id as usize).map_err(cuda_error)?;
    // Torch's current stream, which Torch keeps owning.
    let stream = unsafe {
        let raw = TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device_id);
        Stream::borrow_raw(raw, &ctx)
    };

    // SAFETY: the caller keeps `x` and `out` alive and orders accesses to them
    // through Torch's current stream, on which the kernel is queued.
    unsafe {
        let x = borrow_tensor(&x, &ctx)?;
        let out = borrow_tensor(&out, &ctx)?;
        kernels::relu(out.partition([TILE]), x).async_on(&stream)
    }
    .map_err(cuda_error)?;

    Ok(())
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(relu_rust_cutile, relu_rust_cutile);
