use cutile::prelude::{IntoPartition, Reshape};
use kernels_cutile::{Gpu, cuda_error};
use tvm_ffi::{Error, Result, Tensor, VALUE_ERROR};

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

fn relu_rust_cutile(x: Tensor, out: Tensor) -> Result<()> {
    if x.numel() != out.numel() {
        return Err(Error::new(
            VALUE_ERROR,
            &format!(
                "input and output must have the same number of elements, got {} and {}",
                x.numel(),
                out.numel()
            ),
            "",
        ));
    }
    if x.numel() == 0 {
        return Ok(());
    }

    let gpu = Gpu::new(x.device())?;
    let numel = x.numel();
    // SAFETY: the caller keeps `x` and `out` alive and orders accesses to them
    // through Torch's current stream, on which the kernel is queued.
    unsafe {
        let x = gpu
            .tensor::<f32>(&x)?
            .reshape(&[numel])
            .map_err(cuda_error)?;
        let out = gpu
            .tensor::<f32>(&out)?
            .reshape(&[numel])
            .map_err(cuda_error)?;
        gpu.launch(kernels::relu(out.partition([TILE]), x))?;
    }

    Ok(())
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(relu_rust_cutile, relu_rust_cutile);
