import time
from d3sim.csrc.inliner import INLINER 
import torch 
import numpy as np 
import pccm 
from tensorpc.flow import observe_function
from cumm.inliner import torch_tensor_to_tv
from cumm import tensorview as tv 

@observe_function
def downsample_indices(points: torch.Tensor, voxel_size: float, hash_scale: float = 1.3):
    """Downsample points, we convert points to unsigned int with 20bit precision,
    so value of points must in range (-2 ** 20 * voxel_size, 2 ** 20 * voxel_size),
    if voxel_size is 0.05, the range is (-1048576, 1048576).
    if voxel_size is 0.01, the range is (-209715, 209715).

    WARNING: in apple silicon, the range of z is (-2 ** 19 * voxel_size, 2 ** 19 * voxel_size)
        due to apple silicon don't support 64bit atomic cas, we can only use 62bit key, not 64bit.
    """
    hash_length = int(1.3 * points.shape[0])
    hashkeys_th = torch.empty((hash_length,), dtype=torch.int64, device=points.device)
    hashvalues = torch.empty((hash_length,), dtype=torch.int32, device=points.device)
    points_stride = points.stride(0)
    hashkeys = torch_tensor_to_tv(hashkeys_th, dtype=tv.uint64)
    cnt = torch.zeros((1,), dtype=torch.int32, device=points.device)
    res_indices = torch.empty((points.shape[0],), dtype=torch.int32, device=points.device)
    
    with INLINER.enter_inliner_scope():
        hash_table_t = "tv::hash::LinearHashTableSplit<uint64_t, int, tv::hash::SpatialHash<uint64_t>>;"
        INLINER.kernel_1d("clear_hash", hashkeys.shape[0], 0, f"""
        using table_t = {hash_table_t};
        $hashkeys[i] = table_t::empty_key;
        """)
        code = pccm.code()
        code.raw(f"""
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        using table_t = {hash_table_t};
        table_t table($hashkeys, $hashvalues, $hash_length);
        auto stride = $points_stride;
        auto vsize = $voxel_size;
        uint32_t xc, yc, zc;
        xc = math_op_t::floor(($points[i * stride + 0]) / (vsize)) + (1u << 20);
        yc = math_op_t::floor(($points[i * stride + 1]) / (vsize)) + (1u << 20);
        zc = math_op_t::floor(($points[i * stride + 2]) / (vsize)) + (1u << 20);
        table.insert({{xc, yc, zc}}, int(i));
        """)
        INLINER.kernel_1d(f"insert_table", points.shape[0], 0, code)

        INLINER.kernel_1d("collect_unique_res", hashkeys.shape[0], 0, f"""
        using table_t = {hash_table_t};
        if ($hashkeys[i] != table_t::empty_key){{
            auto old = tv::parallel::atomicAggInc($cnt);
            $res_indices[old] = $hashvalues[i];
        }}
        """)
    cnt_cpu = cnt.cpu().numpy()
    return res_indices[:cnt_cpu[0]]