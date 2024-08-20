import time
from d3sim.constants import D3SIM_DEFAULT_DEVICE, IsAppleSiliconMacOs
from d3sim.csrc.inliner import INLINER 
import torch 
import numpy as np 
from d3sim.algos.d3gs.data import scene
import pccm 
from tensorpc.flow import observe_function
from cumm.inliner import torch_tensor_to_tv
from cumm import tensorview as tv 
from cumm.inliner import torch_tensor_to_tv, measure_and_print_torch

from cumm.dtypes import get_dtype_from_tvdtype
def simple_knn(points: torch.Tensor, k: int, use_64bit: bool = False):
    """Simple KNN, idea come from https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting 
    only 20x faster than cpu in apple m3.
    """
    assert points.is_contiguous()
    num_point = points.shape[0]
    point_stride = points.stride(0)
    points_min, _ = points[:, :3].min(0)
    points_max, _ = points[:, :3].max(0)
    morton_code = torch.empty(num_point, dtype=torch.int64 if use_64bit else torch.int32, device=points.device )
    morton_code_tv_uint = torch_tensor_to_tv(morton_code, dtype=tv.uint64 if use_64bit else tv.uint32)
    code = pccm.code()
    cpp_dtype = get_dtype_from_tvdtype(morton_code_tv_uint.dtype)
    shift_bit = 21 if use_64bit else 10
    box_size = 256
    num_box = tv.div_up(num_point, box_size)
    boxes_minmax = torch.empty([num_box, 6], dtype=torch.float32, device=points.device)
    # with measure_and_print_torch("1"):
    INLINER.kernel_1d(f"to_morton_{shift_bit}", num_point, 0, f"""
    namespace op = tv::arrayops;
    using math_op_t = tv::arrayops::MathScalarOp<float>;

    auto point_max = op::reinterpret_cast_array_nd<3>($points_max)[0];
    auto point_min = op::reinterpret_cast_array_nd<3>($points_min)[0];
    auto point = op::reinterpret_cast_array_nd<3>($points + i * $point_stride)[0];
    auto relative_point = (point - point_min) / (point_max - point_min);
    auto relative_point_scaled = (relative_point * ((1u << {shift_bit}) - 1u)).cast<uint32_t>();
    auto morton = tv::hash::Morton<{cpp_dtype}>::encode(relative_point_scaled[0], relative_point_scaled[1], relative_point_scaled[2]);
    $morton_code_tv_uint[i] = morton;
    """)
    indices = morton_code.argsort(dim=0)
    launch_param = tv.LaunchParam((num_box, 1, 1), (box_size, 1, 1))
    code = pccm.code()
    if IsAppleSiliconMacOs:
        code.raw(f"""
        uint thread_rank = threadPositionInThreadgroup.x;
        uint i = thread_rank + threadgroupPositionInGrid.x * {box_size};
        uint box_id = threadgroupPositionInGrid.x;
        """)
    else:
        code.raw(f"""
        uint32_t thread_rank = threadIdx.x;
        uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t box_id = blockIdx.x;
        """)

    code.raw(f"""  
    namespace op = tv::arrayops;
    using math_op_t = tv::arrayops::MathScalarOp<float>;
    tv::array_nd<float, 2, 3> cur_minmax;
    if (i < $num_point){{
        auto indice = $indices[i];
        auto point = op::reinterpret_cast_array_nd<3>($points + indice * $point_stride)[0];
        cur_minmax[0] = point;
        cur_minmax[1] = point;
    }}else{{
        cur_minmax[1] = {{
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest()
        }};
        cur_minmax[0] = {{
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()
        }};
    }}
    auto warp_idx = thread_rank / 32;
    constexpr auto num_warp = {box_size} / 32;
    TV_SHARED_MEMORY tv::array_nd<float, num_warp> shared_minmax[6];
    cur_minmax[0] = op::apply(tv::parallel::warp_min<float>, cur_minmax[0]);
    cur_minmax[1] = op::apply(tv::parallel::warp_max<float>, cur_minmax[1]);

    if (warp_idx < num_warp){{
        shared_minmax[0][warp_idx] = cur_minmax[0][0];
        shared_minmax[1][warp_idx] = cur_minmax[0][1];
        shared_minmax[2][warp_idx] = cur_minmax[0][2];
        shared_minmax[3][warp_idx] = cur_minmax[1][0];
        shared_minmax[4][warp_idx] = cur_minmax[1][1];
        shared_minmax[5][warp_idx] = cur_minmax[1][2];
    }}
    tv::parallel::block_sync_shared_io();
    if (thread_rank < 32){{
        // do final reduce on one warp
        if (thread_rank < num_warp){{
            cur_minmax[0][0] = shared_minmax[0][thread_rank];
            cur_minmax[0][1] = shared_minmax[1][thread_rank];
            cur_minmax[0][2] = shared_minmax[2][thread_rank];
            cur_minmax[1][0] = shared_minmax[3][thread_rank];
            cur_minmax[1][1] = shared_minmax[4][thread_rank];
            cur_minmax[1][2] = shared_minmax[5][thread_rank];
        }}else{{
            cur_minmax[1] = {{
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest()
            }};
            cur_minmax[0] = {{
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()
            }};
        }}
        cur_minmax[0] = op::apply(tv::parallel::warp_min<float>, cur_minmax[0]);
        cur_minmax[1] = op::apply(tv::parallel::warp_max<float>, cur_minmax[1]);
    }}
    if (thread_rank == 0){{
        op::reinterpret_cast_array_nd<2, 3>($boxes_minmax)[box_id] = cur_minmax;
    }}
    """)
    INLINER.kernel_raw(f"block_reduce_get_box_minmax", launch_param, code)
    res_knn = torch.empty([num_point, k], dtype=torch.float32, device=points.device)

    code = pccm.code()
    code.raw(f"""
    namespace op = tv::arrayops;
    using math_op_t = tv::arrayops::MathScalarOp<float>;
    auto indice = $indices[i];
    auto point = op::reinterpret_cast_array_nd<3>($points + indice * $point_stride)[0];
    tv::array<float, {k}> best = op::constant_array<float, {k}>(std::numeric_limits<float>::max());

    for (int j = std::max(0, int(i) - {k}); j <= std::min(int($num_point) - 1, int(i) + {k}); ++j){{
        if (j == i){{
            continue;
        }}
        auto cur_point = op::reinterpret_cast_array_nd<3>($points + $indices[j] * $point_stride)[0];
        auto dist_square = (cur_point - point).op<op::length2>();
        #pragma unroll
        for (int k = 0; k < {k}; ++k){{
            if (best[k] > dist_square){{
                float t = best[k];
                best[k] = dist_square;
                dist_square = t;
            }}
        }}
    }}
	float reject = best[{k} - 1];
    best = op::constant_array<float, {k}>(std::numeric_limits<float>::max());
    for (int b = 0; b < $num_box; ++b){{
        auto box_minmax = op::reinterpret_cast_array_nd<2, 3>($boxes_minmax)[b];
        tv::array<float, 3> diff{{}};
        if (point[0] < box_minmax[0][0] || point[0] > box_minmax[1][0])
            diff[0] = math_op_t::min(math_op_t::abs(point[0] - box_minmax[0][0]), math_op_t::abs(point[0] - box_minmax[1][0]));
        if (point[1] < box_minmax[0][1] || point[1] > box_minmax[1][1])
            diff[1] = math_op_t::min(math_op_t::abs(point[1] - box_minmax[0][1]), math_op_t::abs(point[1] - box_minmax[1][1]));
        if (point[2] < box_minmax[0][2] || point[2] > box_minmax[1][2])
            diff[2] = math_op_t::min(math_op_t::abs(point[2] - box_minmax[0][2]), math_op_t::abs(point[2] - box_minmax[1][2]));
        auto dist_square = diff.op<op::length2>();
		if (dist_square > reject || dist_square > best[{k} - 1])
			continue;
        for (int j = b * {box_size}; j < std::min((b + 1) * {box_size}, int($num_point)); ++j){{
            if (j == i){{
                continue;
            }}
            auto cur_point = op::reinterpret_cast_array_nd<3>($points + $indices[j] * $point_stride)[0];
            auto dist_square = (cur_point - point).op<op::length2>();
            #pragma unroll
            for (int k = 0; k < {k}; ++k){{
                if (best[k] > dist_square){{
                    float t = best[k];
                    best[k] = dist_square;
                    dist_square = t;
                }}
            }}
        }}
        op::reinterpret_cast_array_nd<{k}>($res_knn)[indice] = op::apply(math_op_t::sqrt, best);
    }}
    """)
    INLINER.kernel_1d(f"do_knn_{k}_{box_size}", num_point, 0, code)
    return res_knn

def __main():
    from d3sim.algos.d3gs.data.load import load_scene_info_and_first_cam
    path =  "/Users/yanyan/Downloads/360_v2/garden"
    scene_info, cam = load_scene_info_and_first_cam(path)
    points = scene_info.point_cloud
    print(points.points.shape)
    points_th = torch.from_numpy(points.points).to(D3SIM_DEFAULT_DEVICE).contiguous()
    def knn_ref(x: torch.Tensor, K: int = 4) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors

        x_np = x.cpu().numpy()
        model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
        distances, _ = model.kneighbors(x_np)
        return torch.from_numpy(distances).to(x)

    res_knn = simple_knn(points_th, 6, False)
    for i in range(5):
        torch.mps.synchronize()
        t = time.time()
        res_knn = simple_knn(points_th, 6, False)
        torch.mps.synchronize()
        print(time.time() - t)
    t = time.time()

    ref_knn = knn_ref(points_th, 7)[:, 1:]
    print("knn_ref", time.time() - t)

    print(res_knn.shape, ref_knn.shape)
    print(torch.linalg.norm(res_knn - ref_knn))
    breakpoint()

if __name__ == "__main__":
    __main() 