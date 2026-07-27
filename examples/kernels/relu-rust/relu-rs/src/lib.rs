use tvm_ffi::{Result, Tensor};

/// ReLU on contiguous float32 CPU tensors.
fn relu_rust(x: Tensor, out: Tensor) -> Result<()> {
    let x_data = x.data_as_slice::<f32>()?;
    let out_data = out.data_as_slice_mut::<f32>()?;

    for (out_elem, &x_elem) in out_data.iter_mut().zip(x_data.iter()) {
        *out_elem = x_elem.max(0.0);
    }

    Ok(())
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(relu_rust, relu_rust);
