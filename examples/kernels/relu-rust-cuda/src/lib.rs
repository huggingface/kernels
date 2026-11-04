use cuda_core::simt::LaunchConfig;
use cuda_device::cuda_module;
use kernels_cuda::cuda_error;
use tvm_ffi::{Error, Result, Tensor, VALUE_ERROR};

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};

    #[kernel]
    pub fn relu(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let v = x[i];
            *slot = if v > 0.0 { v } else { 0.0 };
        }
    }
}

fn relu_rust_cuda(x: Tensor, out: Tensor) -> Result<()> {
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

    let gpu = kernels_cuda::load(x.device(), kernels::from_module)?;
    let x = gpu.view::<f32>(&x)?;
    let mut out = gpu.view::<f32>(&out)?;
    let config = LaunchConfig::for_num_elems(x.len() as u32);
    unsafe { gpu.module.relu(&gpu.stream, config, &x, &mut out) }.map_err(cuda_error)
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(relu_rust_cuda, relu_rust_cuda);
