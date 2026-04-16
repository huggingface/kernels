use std::io::Write;

use eyre::{bail, Context, Result};
use itertools::Itertools;
use kernels_data::config::{Backend, Build, General, TvmFfi};
use minijinja::{context, Environment};

use crate::pyproject::common::{
    prefix_and_join_includes, write_add_build_metadata_py, write_cmake_file, write_compat_py,
    write_metadata,
};
use crate::pyproject::deps::render_deps;
use crate::pyproject::kernel::render_kernel_components;
use crate::pyproject::ops_identifier::KernelIdentifier;
use crate::pyproject::FileSet;

static BUILD_VARIANTS_UTILS: &str = include_str!("../templates/tvm_ffi/build-variants.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static OPS_PY_IN: &str = include_str!("../templates/tvm_ffi/_ops.py.in");
static DETECT_CUDA_CAPABILITY_PY: &str =
    include_str!("../templates/tvm_ffi/cuda/detect-cuda-capability.py");

fn write_cmake_helpers(file_set: &mut FileSet) {
    write_cmake_file(file_set, "utils.cmake", CMAKE_UTILS.as_bytes());
    write_cmake_file(file_set, "kernel.cmake", CMAKE_KERNEL.as_bytes());
    write_cmake_file(
        file_set,
        "build-variants.cmake",
        BUILD_VARIANTS_UTILS.as_bytes(),
    );
    write_cmake_file(file_set, "_ops.py.in", OPS_PY_IN.as_bytes());
    write_add_build_metadata_py(file_set);
    write_cmake_file(
        file_set,
        "cuda/detect-cuda-capability.py",
        DETECT_CUDA_CAPABILITY_PY.as_bytes(),
    );
}

pub fn write_tvm_ffi_ext(
    env: &Environment,
    build: &Build,
    kernel_id: &KernelIdentifier,
) -> Result<FileSet> {
    let tvm_ffi_ext = match build.framework.tvm_ffi() {
        Some(torch_ext) => torch_ext,
        None => bail!("Build configuration does not have `tvm-ffi` section"),
    };

    let mut file_set = FileSet::default();

    write_cmake(
        env,
        build,
        tvm_ffi_ext,
        build.general.name.as_str(),
        kernel_id,
        &mut file_set,
    )?;

    write_setup_py(env, &build.general, tvm_ffi_ext, &kernel_id, &mut file_set)?;

    write_compat_py(&mut file_set)?;

    write_pyproject_toml(env, &build.general, &mut file_set)?;

    write_metadata(&build.general, kernel_id, &mut file_set)?;

    Ok(file_set)
}

pub fn write_setup_py(
    env: &Environment,
    general: &General,
    tvm_ffi: &TvmFfi,
    kernel_id: &KernelIdentifier,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = tvm_ffi
        .data_extensions()
        .map(|exts| exts.iter().map(|ext| format!("\"**/*.{ext}\"")).join(", "));

    env.get_template("tvm_ffi/setup.py")
        .wrap_err("Cannot get tvm_ffi setup.py template")?
        .render_captured_to(
            context! {
                data_globs => data_globs,
                kernel_name => kernel_id.name(),
                kernel_unique_id => kernel_id.unique_id(),
                python_name => general.name.python_name(),
            },
            writer,
        )
        .wrap_err("Cannot render tvm_ffi setup.py template")?;

    Ok(())
}

pub fn write_pyproject_toml(
    env: &Environment,
    general: &General,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("pyproject.toml");

    // Common python dependencies (no backend-specific ones)
    let python_dependencies = itertools::process_results(general.python_depends(), |iter| {
        iter.flat_map(|(_, deps)| deps.python.iter().map(|d| format!("\"{}\"", d.pkg)))
            .join(", ")
    })?;

    // Collect backend-specific dependencies for all backends
    let mut backend_dependencies = Vec::new();
    for backend in &Backend::all() {
        let deps = itertools::process_results(general.backend_python_depends(*backend), |iter| {
            iter.flat_map(|(_, deps)| deps.python.iter().map(|d| format!("\"{}\"", d.pkg)))
                .join(", ")
        })?;

        if !deps.is_empty() {
            backend_dependencies.push((backend.to_string(), deps));
        }
    }

    env.get_template("tvm_ffi/pyproject.toml")
        .wrap_err("Cannot get tvm_ffi pyproject.toml template")?
        .render_captured_to(
            context! {
                python_name => general.name.python_name(),
                python_dependencies => python_dependencies,
                backend_dependencies => backend_dependencies,
            },
            writer,
        )
        .wrap_err("Cannot render tvm_ffi pyproject.toml template")?;

    Ok(())
}

pub fn render_binding(
    env: &Environment,
    tvm_ffi: &TvmFfi,
    name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("tvm_ffi/binding.cmake")
        .wrap_err("Cannot get tvm_ffi binding template")?
        .render_captured_to(
            context! {
                includes => tvm_ffi.include.as_ref().map(prefix_and_join_includes),
                name => name,
                src => tvm_ffi.src,
            },
            &mut *write,
        )
        .wrap_err("Cannot render tvm_ffi binding template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn render_extension(
    env: &Environment,
    general: &General,
    tvm_ffi: &TvmFfi,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("tvm_ffi/tvm-ffi-extension.cmake")
        .wrap_err("Cannot get tvm_ffi extension template")?
        .render_captured_to(
            context! {
                python_name => general.name.python_name(),
                data_extensions => tvm_ffi.data_extensions(),
            },
            &mut *write,
        )
        .wrap_err("Cannot render tvm_ffi extension template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn render_preamble(
    env: &Environment,
    general: &General,
    kernel_id: &KernelIdentifier,
    write: &mut impl Write,
) -> Result<()> {
    let cuda_minver = general.cuda.as_ref().and_then(|c| c.minver.as_ref());
    let cuda_maxver = general.cuda.as_ref().and_then(|c| c.maxver.as_ref());

    env.get_template("tvm_ffi/preamble.cmake")
        .wrap_err("Cannot get tvm_ffi preamble template")?
        .render_captured_to(
            context! {
                kernel_name => kernel_id.name(),
                kernel_unique_id => kernel_id.unique_id(),
                name => &general.name,
                python_name => general.name.python_name(),
                cuda_minver => cuda_minver.map(|v| v.to_string()),
                cuda_maxver => cuda_maxver.map(|v| v.to_string()),
            },
            &mut *write,
        )
        .wrap_err("Cannot render tvm_ffi preamble template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn write_cmake(
    env: &Environment,
    build: &Build,
    tvm_ffi: &TvmFfi,
    name: &str,
    kernel_id: &KernelIdentifier,
    file_set: &mut FileSet,
) -> Result<()> {
    write_cmake_helpers(file_set);

    let cmake_writer = file_set.entry("CMakeLists.txt");

    render_preamble(env, &build.general, kernel_id, cmake_writer)?;

    render_deps(env, build, cmake_writer)?;

    render_binding(env, tvm_ffi, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, &build.general, tvm_ffi, cmake_writer)?;

    Ok(())
}
