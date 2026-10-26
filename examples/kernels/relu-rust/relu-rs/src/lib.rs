use tvm_ffi::{Error, Result, Tensor, VALUE_ERROR};

fn relu_rust(x: Tensor, out: Tensor) -> Result<()> {
    let x_data = x.data_as_slice::<f32>()?;
    let out_data = out.data_as_slice_mut::<f32>()?;

    // `zip` would otherwise stop at the shorter slice, leaving the tail of a
    // larger `out` holding whatever `empty_like` allocated.
    if x_data.len() != out_data.len() {
        return Err(Error::new(
            VALUE_ERROR,
            &format!(
                "input and output must have the same number of elements, got {} and {}",
                x_data.len(),
                out_data.len()
            ),
            "",
        ));
    }

    for (out_elem, &x_elem) in out_data.iter_mut().zip(x_data.iter()) {
        *out_elem = x_elem.max(0.0);
    }

    Ok(())
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(relu_rust, relu_rust);
