import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC
from d3sim.core.geodef import EulerIntrinsicOrder

from d3sim.core import dataclass_dispatch as dataclasses

class SHConstants:
    SH_C0: float = 0.28209479177387814
    SH_C1: float = 0.4886025119029199
    SH_C2: list[float] = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    SH_C3: list[float] = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    

class GLMDebugDep(pccm.Class):
    def __init__(self):
        super().__init__()
        self.build_meta.add_public_includes("/Users/yanyan/Projects/glm")
        self.add_include("glm/glm.hpp")

class Gaussian3D(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC, GLMDebugDep)
    
    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def sh_dir_to_rgb(self):
        code = pccm.code()
        code.nontype_targ("Degree", "int")
        code.targ("T")
        code.arg("sh_ptr", "device const tv::array<T, 3>*")
        code.arg("dir", "tv::array<T, 3>")
        code.raw(f"""
        namespace op = tv::arrayops;
        op::PointerValueReader<device const tv::array<T, 3>> sh(sh_ptr);
        static_assert(Degree >= 0 && Degree <= 3, "Degree must be in [0, 3]");
        tv::array<T, 3> result = T({SHConstants.SH_C0}) * sh[0];
        if (Degree > 0)
        {{
            T x = dir[0];
            T y = dir[1];
            T z = dir[2];
            result = result - {SHConstants.SH_C1} * y * sh[1] + T({SHConstants.SH_C1}) * z * sh[2] - T({SHConstants.SH_C1}) * x * sh[3];

            if (Degree > 1)
            {{
                T xx = x * x, yy = y * y, zz = z * z;
                T xy = x * y, yz = y * z, xz = x * z;
                result = result +
                    T({SHConstants.SH_C2[0]}) * xy * sh[4] +
                    T({SHConstants.SH_C2[1]}) * yz * sh[5] +
                    T({SHConstants.SH_C2[2]}) * (T(2) * zz - xx - yy) * sh[6] +
                    T({SHConstants.SH_C2[3]}) * xz * sh[7] +
                    T({SHConstants.SH_C2[4]}) * (xx - yy) * sh[8];

                if (Degree > 2)
                {{
                    result = result +
                        T({SHConstants.SH_C3[0]}) * y * (T(3) * xx - yy) * sh[9] +
                        T({SHConstants.SH_C3[1]}) * xy * z * sh[10] +
                        T({SHConstants.SH_C3[2]}) * y * (T(4) * zz - xx - yy) * sh[11] +
                        T({SHConstants.SH_C3[3]}) * z * (T(2) * zz - T(3) * xx - T(3) * yy) * sh[12] +
                        T({SHConstants.SH_C3[4]}) * x * (T(4) * zz - xx - yy) * sh[13] +
                        T({SHConstants.SH_C3[5]}) * z * (xx - yy) * sh[14] +
                        T({SHConstants.SH_C3[6]}) * x * (xx - T(3) * yy) * sh[15];
                }}
            }}
        }}
        result += T(0.5);
        return result;
        """)
        return code.ret("tv::array<T, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def scale_quat_to_cov3d(self):
        code = pccm.code()
        code.targ("T")
        code.arg("scale", "tv::array<T, 3>")
        code.arg("scale_global", "T")
        code.arg("quat", "tv::array<T, 4>")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto S = (scale * scale_global).template op<op::from_diagonal>();
        auto R = quat.template op<op::uqmat_colmajor>();
        auto M = S.template op<op::mm_nn>(R);
        auto sigma = M.template op<op::transpose>().template op<op::mm_nn>(M);
        return {{sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1], sigma[1][2], sigma[2][2]}};
        """)
        return code.ret("tv::array<T, 6>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def scale_quat_to_cov3d_glm_ref(self):
        code = pccm.code()
        code.targ("T")
        code.arg("scale", "tv::array<T, 3>")
        code.arg("scale_global", "T")
        code.arg("quat", "tv::array<T, 4>")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto S = (scale * scale_global).template op<op::from_diagonal>();
        auto R = quat.template op<op::uqmat_colmajor>();
        auto M = S.template op<op::mm_nn>(R);
        auto sigma = M.template op<op::transpose>().template op<op::mm_nn>(M);
        return {{sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1], sigma[1][2], sigma[2][2]}};
        """)
        return code.ret("tv::array<T, 6>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def project_gaussian_to_2d(self):
        code = pccm.code()
        code.targ("T")
        code.arg("mean_camera", "tv::array<T, 3>")
        code.arg("focal_length, tan_fov", "tv::array<T, 2>")
        code.arg("cam2world_T", "tv::array_nd<T, 4, 3>")
        code.arg("cov3d_vec", "tv::array_nd<T, 6>")

        code.arg("clamp_factor", "T", "1.3f")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto limit = tan_fov * clamp_factor * mean_camera[2];
        auto txylimit = op::slice<0, 2>(mean_camera).template op<op::clamp>(-limit, limit);
        auto txylimit_focal = txylimit * focal_length;
        auto tz_square = mean_camera[2] * mean_camera[2];

        // we need T = W.T @ J.T, if @ is colmajor, we need 
        // W.T stored as colmajor and J.T stored as colmajor,
        // so store W as rowmajor and J as rowmajor.
        // J_T_cm = J_rm
        tv::array_nd<T, 3, 3> J_T_cm{{
            tv::array<T, 3>{{focal_length[0] / mean_camera[2], 0, -txylimit_focal[0] / tz_square}},
            tv::array<T, 3>{{0, focal_length[1] / mean_camera[2], -txylimit_focal[1] / tz_square}},
            tv::array<T, 3>{{0, 0, 0}},
        }};
        // world2cam_T = cam2world_T.inverse() = cam2world_T.T
        // so world2cam_T_cm = cam2world_T
        // auto world2cam_T = op::slice<0, 3>(cam2world_T).template op<op::transpose>();
        auto world2cam_T_cm = op::slice<0, 3>(cam2world_T);
        
        auto W_T_matmul_J_T_cm = world2cam_T_cm.template op<op::mm_nn>(J_T_cm);
        tv::array_nd<T, 3, 3> Vrk{{
            tv::array<T, 3>{{cov3d_vec[0], cov3d_vec[1], cov3d_vec[2]}},
            tv::array<T, 3>{{cov3d_vec[1], cov3d_vec[3], cov3d_vec[4]}},
            tv::array<T, 3>{{cov3d_vec[2], cov3d_vec[4], cov3d_vec[5]}}
        }};
        // Vrk.T = Vrk
        // Projected Cov = J @ W @ Cov @ W.T @ J.T
        // = W_T_matmul_J_T_cm.T @ Vrk @ W_T_matmul_J_T_cm
        // auto cov_projected = W_T_matmul_J_T_cm.template op<op::transpose>().template op<op::mm_nn>(Vrk.op<op::transpose>()).op<op::mm_nn>(W_T_matmul_J_T_cm);
        
        auto cov_projected = W_T_matmul_J_T_cm.template op<op::mm_tn>(Vrk).template op<op::mm_nn>(W_T_matmul_J_T_cm);
        return {{cov_projected[0][0], cov_projected[0][1], cov_projected[1][1]}};
        """)
        return code.ret("tv::array<T, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gaussian_2d_inverse_and_det(self):
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_vec", "tv::array<T, 3>")
        code.arg("eps", "T", "1e-6")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        T det = cov2d_vec[0] * cov2d_vec[2] - cov2d_vec[1] * cov2d_vec[1];
        T det_inv = T(1) / (det + eps);
        tv::array<T, 3> ret{{
            cov2d_vec[2] * det_inv,
            -cov2d_vec[1] * det_inv,
            cov2d_vec[0] * det_inv,
        }};
        return std::make_tuple(ret, det);
        """)
        return code.ret("std::tuple<tv::array<T, 3>, T>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def get_gaussian_2d_ellipse(self):
        # https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_vec", "tv::array<T, 3>")
        code.arg("det", "T")
        code.arg("ellipsis_eps", "T", "1e-1")
        code.arg("std_scale", "T", "3")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;

        T mid = T(0.5) * (cov2d_vec[0] + cov2d_vec[3]);
        T sqrt_part = math_op_t::sqrt(math_op_t::max(ellipsis_eps, mid * mid - det));
        T eigen1 = mid + sqrt_part;
        T eigen2 = mid - sqrt_part;
        // ellipsis major radius equal to sqrt(max(eigen1, eigen2))
        // to expand to 99% of the gaussian, use 3 times of the radius
        // 3 sigma rule
        T major_radius = math_op_t::sqrt(math_op_t::max(eigen1, eigen2));
        return std_scale * major_radius;
        """)
        return code.ret("T")

    # @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    # def get_gaussian_2d_block_rect_xyxy(self):
    #     # https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    #     code = pccm.code()
    #     code.nontype_targ("BlockX", "int")
    #     code.nontype_targ("BlockY", "int")

    #     code.targ("T")

    #     code.arg("cov2d_vec", "tv::array_nd<T, 3, 3>")
    #     code.arg("grid_xy", "tv::array<int, 2>")

    #     code.arg("det", "T")
    #     code.arg("ellipsis_eps", "T", "1e-1")
    #     code.arg("std_scale", "T", "3")

    #     code.raw(f"""
    #     namespace op = tv::arrayops;
    #     using math_op_t = tv::arrayops::MathScalarOp<T>;

    #     T mid = T(0.5) * (cov2d_vec[0] + cov2d_vec[3]);
    #     T sqrt_part = math_op_t::sqrt(math_op_t::max(ellipsis_eps, mid * mid - det));
    #     T eigen1 = mid + sqrt_part;
    #     T eigen2 = mid - sqrt_part;
    #     // ellipsis major radius equal to sqrt(max(eigen1, eigen2))
    #     // to expand to 99% of the gaussian, use 3 times of the radius
    #     // 3 sigma rule
    #     T major_radius = math_op_t::sqrt(math_op_t::max(eigen1, eigen2));
    #     return std_scale * major_radius;
    #     """)
    #     return code.ret("T")
