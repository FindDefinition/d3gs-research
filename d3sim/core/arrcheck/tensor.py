from types import EllipsisType, UnionType
from typing import Annotated, Any, Generic, Literal, Sequence, TypeAlias, TypeVar, Union, get_args, get_type_hints, Type
import numpy as np 
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.arrcheck.dtypes import DType, Float32, Float64, get_dtype_to_np_dtype, get_dtype_to_th_dtype
from pydantic import (
    AfterValidator,
)
import dataclasses as original_dataclasses
import torch 

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class ArrayCheckBase:
    shape: Sequence[int | str | EllipsisType]
    dtypes: DType
    device: Literal["cpu", "gpu"] | None = None

def check_shape_of_meta_shape(meta_shapes: Sequence[Sequence[int | str | EllipsisType]], real_shapes: Sequence[Sequence[int]]):
    symbol_to_val: dict[str, tuple[int, Sequence[int | str | EllipsisType]]] = {}
    assert len(meta_shapes) == len(real_shapes)
    for meta_shape, real_shape in zip(meta_shapes, real_shapes):
        shape_before_ellip: Sequence[int | str] = []
        shape_after_ellip: Sequence[int | str] = []
        ellip_found = False

        for elem in meta_shape:
            if elem == Ellipsis:
                if ellip_found:
                    raise ValueError("Ellipsis found more than once")
                ellip_found = True
            if not isinstance(elem, EllipsisType):
                if not ellip_found:
                    shape_before_ellip.append(elem)
                else:
                    shape_after_ellip.append(elem)
        minimum_ndim = len(shape_before_ellip) + len(shape_after_ellip)
        if ellip_found:
            if minimum_ndim > len(real_shape):
                raise ValueError(f"expected ndim >= {minimum_ndim}, got ndim={len(real_shape)}|{real_shape}")
        else:
            if minimum_ndim != len(real_shape):
                raise ValueError(f"expected ndim == {minimum_ndim}, got ndim={len(real_shape)}|{real_shape}")

        for i, s in enumerate(shape_before_ellip):
            if isinstance(s, int) and s >= 0:
                assert s == real_shape[i], f"expected your shape {real_shape}[{i}] == {s}({meta_shape}[{i}])"
            elif isinstance(s, str):
                if s in symbol_to_val:
                    prev_val, defined_shape = symbol_to_val[s]
                    assert prev_val == real_shape[i], f"expected {real_shape}[{i}] == {s} ({prev_val}) defined in previous shape {defined_shape}"
                else:
                    symbol_to_val[s] = (real_shape[i], meta_shape)
        for _i, s in enumerate(shape_after_ellip):
            i = -(len(shape_after_ellip) - _i)
            if isinstance(s, int) and s >= 0:
                assert s == real_shape[i], f"expected {real_shape}[{i}] == {s}|{meta_shape}[{i}]"
            elif isinstance(s, str):
                if s in symbol_to_val:
                    prev_val, defined_shape = symbol_to_val[s]
                    assert prev_val == real_shape[i], f"expected {real_shape}[{i}] == {s} ({prev_val}) defined in previous shape {defined_shape}"
                else:
                    symbol_to_val[s] = (real_shape[i], meta_shape)

def _check_nparray_by_tensor_meta(meta: ArrayCheckBase, arr: np.ndarray):
    found = False
    for dtype in meta.dtypes:
        np_dtype = get_dtype_to_np_dtype(dtype)
        if arr.dtype == np_dtype:
            found = True
            break
    if not found:
        raise ValueError(f"arr.dtype={arr.dtype} is not in {meta.dtypes}")
    check_shape_of_meta_shape([meta.shape], [arr.shape])
    if meta.device:
        assert meta.device == "cpu", "np.ndarray only support cpu"
    return  arr

def _check_thten_by_tensor_meta(meta: ArrayCheckBase, arr: torch.Tensor):
    found = False
    for dtype in meta.dtypes:
        np_dtype = get_dtype_to_th_dtype(dtype)
        if arr.dtype == np_dtype:
            found = True
            break
    if not found:
        raise ValueError(f"arr.dtype={arr.dtype} is not in {meta.dtypes}")
    check_shape_of_meta_shape([meta.shape], [arr.shape])
    if meta.device:
        if meta.device == "gpu":
            assert arr.device.type != "cpu", "expected device tensor"
        else:
            assert arr.device.type == "cpu", "expected cpu tensor"
    return  arr

def _check_arr_or_tensor_by_tensor_meta(meta: ArrayCheckBase, arr: np.ndarray | torch.Tensor | None):
    if arr is None:
        return
    if isinstance(arr, np.ndarray):
        return _check_nparray_by_tensor_meta(meta, arr)
    if isinstance(arr, torch.Tensor):
        return _check_thten_by_tensor_meta(meta, arr)
    raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(arr)}")

@original_dataclasses.dataclass(frozen=True)
class ArrayValidator(AfterValidator):
    meta: ArrayCheckBase 

def ArrayCheck(shape: Sequence[int | str | EllipsisType], dtypes: DType, device: Literal["cpu", "gpu"] | None = None) -> AfterValidator:
    meta = ArrayCheckBase(shape, dtypes, device)
    return ArrayValidator(lambda x: _check_arr_or_tensor_by_tensor_meta(meta, x), meta)


def __main():
    @dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
    class Dev:
        a: Annotated[torch.Tensor, ArrayCheck([-1, 3], Float32 | Float64)]

    meta_shapes = [[1, 2, ..., "N"], [-1, "N"]]
    real_shapes = [[1, 2, 4], [4, 4]]
    check_shape_of_meta_shape(meta_shapes, real_shapes)

    a = Dev(torch.zeros(3, 3))
    for f in dataclasses.fields(a):
        print(f.name, get_type_hints(Dev, include_extras=True))

if __name__ =="__main__":

    __main()