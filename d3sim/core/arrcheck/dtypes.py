
from types import UnionType
from typing import Any, Generic, Literal, Type, TypeAlias
import numpy as np 
from enum import Flag, auto

class DType(Flag):
    Int8 = auto()
    Int16 = auto()
    Int32 = auto()
    Int64 = auto()
    Bfloat16 = auto()
    Float16 = auto()
    Float32 = auto()
    Float64 = auto()
    Bool = auto()
    Uint8 = auto()
    Uint16 = auto()
    Uint32 = auto()
    Uint64 = auto()

Floating = DType.Float16 | DType.Float32 | DType.Float64 | DType.Bfloat16

Integer = DType.Int8 | DType.Int16 | DType.Int32 | DType.Int64 | DType.Uint8 | DType.Uint16 | DType.Uint32 | DType.Uint64

UnsignedInteger = DType.Uint8 | DType.Uint16 | DType.Uint32 | DType.Uint64

SignedInteger = DType.Int8 | DType.Int16 | DType.Int32 | DType.Int64

Int8 = DType.Int8
Int16 = DType.Int16
Int32 = DType.Int32
Int64 = DType.Int64
Uint8 = DType.Uint8
Uint16 = DType.Uint16
Uint32 = DType.Uint32
Uint64 = DType.Uint64
Float16 = DType.Float16
Float32 = DType.Float32
Float64 = DType.Float64
Bfloat16 = DType.Bfloat16
Bool = DType.Bool


I8 = DType.Int8
I16 = DType.Int16
I32 = DType.Int32
I64 = DType.Int64
U8 = DType.Uint8
U16 = DType.Uint16
U32 = DType.Uint32
U64 = DType.Uint64
F16 = DType.Float16
F32 = DType.Float32
F64 = DType.Float64
BF16 = DType.Bfloat16
BOOL = DType.Bool


DTYPE_TO_NPDTYPE: dict[DType, Any] = {
    F16: np.dtype(np.float16),
    F32: np.dtype(np.float32),
    F64: np.dtype(np.float64),
    I8: np.dtype(np.int8),
    I16: np.dtype(np.int16),
    I32: np.dtype(np.int32),
    I64: np.dtype(np.int64),
    U8: np.dtype(np.uint8),
    U16: np.dtype(np.uint16),
    U32: np.dtype(np.uint32),
    U64: np.dtype(np.uint64),
} 
DTYPE_TO_TORCH_DTYPE: dict[DType, Any] = {}

def get_dtype_to_np_dtype(dtype: DType) -> np.dtype:
    return DTYPE_TO_NPDTYPE[dtype]

def get_dtype_to_th_dtype(dtype: DType) -> Any:
    if not DTYPE_TO_TORCH_DTYPE:
        import torch 

        DTYPE_TO_TORCH_DTYPE.update({
            F32: torch.float32,
            F64: torch.float64,
            F16: torch.float16,
            I32: torch.int32,
            I64: torch.int64,
            I8: torch.int8,
            I16: torch.int16,
            U8: torch.uint8,
            BF16: torch.bfloat16,
        })
        torch_version = torch.__version__.split(".")
        major_version = int(torch_version[0])
        minor_version = int(torch_version[1])
        if (major_version, minor_version) >= (2, 3):
            DTYPE_TO_TORCH_DTYPE.update({
                U16: torch.uint16,
                U32: torch.uint32,
                U64: torch.uint64,
            })
    return DTYPE_TO_TORCH_DTYPE[dtype]

