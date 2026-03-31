use std::ffi::c_void;

pub const DL_CPU: i32 = 1;
pub const DL_CUDA: i32 = 2;
pub const DL_ONEAPI: i32 = 14;

pub const DL_INT: u8 = 0;
pub const DL_UINT: u8 = 1;
pub const DL_FLOAT: u8 = 2;
pub const DL_BFLOAT: u8 = 4;

#[repr(C)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

const TVMFFI_DLTENSOR_PTR: i32 = 7;

#[repr(C)]
pub struct TVMFFIAny {
    pub type_index: i32,
    pub zero_padding: u32,
    pub v_ptr: *mut c_void,
}

impl TVMFFIAny {
    pub fn from_dltensor(tensor: *mut DLTensor) -> Self {
        Self {
            type_index: TVMFFI_DLTENSOR_PTR,
            zero_padding: 0,
            v_ptr: tensor as *mut c_void,
        }
    }

    pub fn none() -> Self {
        Self {
            type_index: 0,
            zero_padding: 0,
            v_ptr: std::ptr::null_mut(),
        }
    }
}

// Every `__tvm_ffi_*` export has this signature.
pub type TVMFFIFunc =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;
