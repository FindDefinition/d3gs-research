
import importlib.util 
from pathlib import Path
from typing import List, Optional 

from ccimport import compat
from d3sim.constants import IsAppleSiliconMacOs
import pccm 

from cumm.common import CUDALibs, TensorView
from cumm.gemm.codeops import dispatch
from cumm import dtypes
class DeviceSortIncludes(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("cub/cub.cuh")
        self.add_include("cub/device/device_radix_sort.cuh")


class DeviceSort(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, CUDALibs)

    @pccm.pybind.mark 
    @pccm.cuda.static_function(impl_file_suffix=".cc" if IsAppleSiliconMacOs else ".cu")
    def get_sort_workspace_size(self):
        code = pccm.code()
        code.arg("keys_in, keys_out, values_in, values_out", "tv::Tensor")
        if IsAppleSiliconMacOs:
            code.raw(f"""
            
            TV_THROW_RT_ERR("Not implemented on Apple Silicon");
            """)
        else:
            code.add_dependency(DeviceSortIncludes)

            code.raw(f"""
            size_t sort_size;
            """)
            for key_dtype in dispatch(code, [dtypes.uint32, dtypes.uint64, dtypes.int32, dtypes.int64], "keys_in.dtype()"):
                for val_dtype in dispatch(code, [dtypes.float32, dtypes.uint32, dtypes.uint64, dtypes.int32, dtypes.int64], "values_in.dtype()"):

                    code.raw(f"""
                    cub::DeviceRadixSort::SortPairs(
                        nullptr, sort_size,
                        keys_in.data_ptr<const {key_dtype}>(), 
                        keys_out.data_ptr<{key_dtype}>(),
                        values_in.data_ptr<const {val_dtype}>(), 
                        values_out.data_ptr<{val_dtype}>(), keys_in.dim(0));
                    """)
            code.raw(f"""
            return sort_size;
            """)
        return code.ret("size_t")

    @pccm.pybind.mark 
    @pccm.cuda.static_function(impl_file_suffix=".cc" if IsAppleSiliconMacOs else ".cu")
    def do_radix_sort_with_bit_range(self):
        code = pccm.code()
        code.arg("keys_in, keys_out, values_in, values_out", "tv::Tensor")
        code.arg("workspace", "tv::Tensor")
        code.arg("bit_start, bit_end", "int")
        code.arg("stream_int", "std::uintptr_t")
        if IsAppleSiliconMacOs:
            code.raw(f"""
            
            TV_THROW_RT_ERR("Not implemented on Apple Silicon");
            """)
        else:
            code.add_dependency(DeviceSortIncludes)
            code.raw(f"""
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);
            size_t storage_size = workspace.raw_size();
            """)
            for key_dtype in dispatch(code, [dtypes.uint32, dtypes.uint64, dtypes.int32, dtypes.int64], "keys_in.dtype()"):
                for val_dtype in dispatch(code, [dtypes.float32, dtypes.uint32, dtypes.uint64, dtypes.int32, dtypes.int64], "values_in.dtype()"):

                    code.raw(f"""
                    cub::DeviceRadixSort::SortPairs(
                        workspace.raw_data(), storage_size,
                        keys_in.data_ptr<const {key_dtype}>(), 
                        keys_out.data_ptr<{key_dtype}>(),
                        values_in.data_ptr<const {val_dtype}>(), 
                        values_out.data_ptr<{val_dtype}>(), size_t(keys_in.dim(0)),
                        bit_start, bit_end, stream);
                    """)
        return code 

    @pccm.pybind.mark 
    @pccm.static_function
    def get_higher_msb(self):
        code = pccm.code()
        code.arg("n", "uint32_t")
        code.raw("""
        uint32_t msb = sizeof(n) * 4;
        uint32_t step = msb;
        while (step > 1)
        {
            step /= 2;
            if (n >> msb)
                msb += step;
            else
                msb -= step;
        }
        if (n >> msb)
            msb++;
        return msb;
        """)
        return code.ret("uint32_t")
