use std::ffi::c_void;

use crate::backend::BackendKind;

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

impl From<BackendKind> for DLDevice {
    fn from(kind: BackendKind) -> Self {
        let device_type = match kind {
            BackendKind::Cpu => DL_CPU,
            BackendKind::Cuda => DL_CUDA,
            BackendKind::Xpu => DL_ONEAPI,
        };

        Self {
            device_type,
            device_id: 0,
        }
    }
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
const TVMFFI_NONE: i32 = 0;
const TVMFFI_INT: i32 = 1;
const TVMFFI_BOOL: i32 = 2;
const TVMFFI_FLOAT: i32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub union TVMFFIValue {
    pub v_ptr: *mut c_void,
    pub v_int64: i64,
    pub v_float64: f64,
    pub v_uint64: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TVMFFIAny {
    pub type_index: i32,
    pub zero_padding: u32,
    // Tagged payload for a DLTensor pointer or scalar argument.
    pub value: TVMFFIValue,
}

impl TVMFFIAny {
    pub fn from_dltensor(tensor: *mut DLTensor) -> Self {
        Self {
            type_index: TVMFFI_DLTENSOR_PTR,
            zero_padding: 0,
            value: TVMFFIValue {
                v_ptr: tensor.cast(),
            },
        }
    }

    pub fn from_int(value: i64) -> Self {
        Self {
            type_index: TVMFFI_INT,
            zero_padding: 0,
            value: TVMFFIValue { v_int64: value },
        }
    }

    pub fn from_bool(value: bool) -> Self {
        Self {
            type_index: TVMFFI_BOOL,
            zero_padding: 0,
            value: TVMFFIValue {
                v_int64: i64::from(value),
            },
        }
    }

    pub fn from_float(value: f64) -> Self {
        Self {
            type_index: TVMFFI_FLOAT,
            zero_padding: 0,
            value: TVMFFIValue { v_float64: value },
        }
    }

    pub fn none() -> Self {
        Self {
            type_index: TVMFFI_NONE,
            zero_padding: 0,
            value: TVMFFIValue {
                v_ptr: std::ptr::null_mut(),
            },
        }
    }
}

// Every `__tvm_ffi_*` export has this signature.
pub type TVMFFIFunc =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;
