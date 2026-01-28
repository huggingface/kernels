mod cpu;
pub use cpu::write_torch_ext_cpu;

mod cuda;
pub use cuda::write_torch_ext_cuda;

pub mod common;

pub mod kernel;

mod metal;
pub use metal::write_torch_ext_metal;

mod ops_identifier;
pub(crate) use ops_identifier::kernel_ops_identifier;

mod noarch;
pub use noarch::write_torch_ext_noarch;

mod xpu;
pub use xpu::write_torch_ext_xpu;
