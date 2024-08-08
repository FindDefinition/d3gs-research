import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC
from d3sim.core.geodef import EulerIntrinsicOrder

from d3sim.core import dataclass_dispatch as dataclasses
from ccimport import compat
class SHConstants:
    C0: float = 0.28209479177387814
    C1: float = 0.4886025119029199
    C2: list[float] = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3: list[float] = [
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
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC)
        if compat.IsAppleSiliconMacOs:
            self.add_code_before_class("""
            constant float SH_C0 = 0.28209479177387814f;
            constant float SH_C1 = 0.4886025119029199f;
            constant float SH_C2[5] = {
                1.0925484305920792f,
                -1.0925484305920792f,
                0.31539156525252005f,
                -1.0925484305920792f,
                0.5462742152960396f
            };
            constant float SH_C3[7] = {
                -0.5900435899266435f,
                2.890611442640554f,
                -0.4570457994644658f,
                0.3731763325901154f,
                -0.4570457994644658f,
                1.445305721320277f,
                -0.5900435899266435f
            };
            """)
        else:
            self.add_code_before_class("""
            __device__ const float SH_C0 = 0.28209479177387814f;
            __device__ const float SH_C1 = 0.4886025119029199f;
            __device__ const float SH_C2[5] = {
                1.0925484305920792f,
                -1.0925484305920792f,
                0.31539156525252005f,
                -1.0925484305920792f,
                0.5462742152960396f
            };
            __device__ const float SH_C3[7] = {
                -0.5900435899266435f,
                2.890611442640554f,
                -0.4570457994644658f,
                0.3731763325901154f,
                -0.4570457994644658f,
                1.445305721320277f,
                -0.5900435899266435f
            };
            """)
    
    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def sh_dir_to_rgb(self):
        code = pccm.code()
        code.nontype_targ("Degree", "int")
        code.targ("T")
        code.arg("dir", "tv::array<T, 3>")
        code.arg("sh_ptr", "TV_METAL_DEVICE const tv::array<T, 3>*")
        code.raw(f"""
        namespace op = tv::arrayops;
        op::PointerValueReader<TV_METAL_DEVICE const tv::array<T, 3>> sh(sh_ptr);
        static_assert(Degree >= 0 && Degree <= 3, "Degree must be in [0, 3]");
        tv::array<T, 3> result = T({SHConstants.C0}) * sh[0];
        if (Degree > 0)
        {{
            T x = dir[0];
            T y = dir[1];
            T z = dir[2];
            constexpr T C1 = T({SHConstants.C1});
            result = result - C1 * y * sh[1] + C1 * z * sh[2] - C1 * x * sh[3];

            if (Degree > 1)
            {{
                T xx = x * x, yy = y * y, zz = z * z;
                T xy = x * y, yz = y * z, xz = x * z;
                result = result +
                    T({SHConstants.C2[0]}) * xy * sh[4] +
                    T({SHConstants.C2[1]}) * yz * sh[5] +
                    T({SHConstants.C2[2]}) * (T(2) * zz - xx - yy) * sh[6] +
                    T({SHConstants.C2[3]}) * xz * sh[7] +
                    T({SHConstants.C2[4]}) * (xx - yy) * sh[8];

                if (Degree > 2)
                {{
                    result = result +
                        T({SHConstants.C3[0]}) * y * (T(3) * xx - yy) * sh[9] +
                        T({SHConstants.C3[1]}) * xy * z * sh[10] +
                        T({SHConstants.C3[2]}) * y * (T(4) * zz - xx - yy) * sh[11] +
                        T({SHConstants.C3[3]}) * z * (T(2) * zz - T(3) * xx - T(3) * yy) * sh[12] +
                        T({SHConstants.C3[4]}) * x * (T(4) * zz - xx - yy) * sh[13] +
                        T({SHConstants.C3[5]}) * z * (xx - yy) * sh[14] +
                        T({SHConstants.C3[6]}) * x * (xx - T(3) * yy) * sh[15];
                }}
            }}
        }}
        result += T(0.5);
        return result;
        """)
        return code.ret("tv::array<T, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def sh_dir_to_rgb_grad(self):
        code = pccm.code()
        code.nontype_targ("Degree", "int")
        code.targ("T")
        code.arg("drgb", "tv::array<T, 3>")
        code.arg("dsh_ptr", "TV_METAL_DEVICE tv::array<T, 3>*")
        code.arg("dir", "tv::array<T, 3>")
        code.arg("sh_ptr", "TV_METAL_DEVICE const tv::array<T, 3>*")
        code.raw(f"""
        namespace op = tv::arrayops;
        op::PointerValueReader<TV_METAL_DEVICE const tv::array<T, 3>> sh(sh_ptr);
        static_assert(Degree >= 0 && Degree <= 3, "Degree must be in [0, 3]");
        tv::array<T, 3> dRGBdx{{}};
        tv::array<T, 3> dRGBdy{{}};
        tv::array<T, 3> dRGBdz{{}};

        T x = dir[0];
        T y = dir[1];
        T z = dir[2];

        dsh_ptr[0] = T(SH_C0) * drgb;
        tv::array<T, 3> result = T(SH_C0) * sh[0];
        if (Degree > 0)
        {{
            constexpr T C1 = T(SH_C1);
            T dRGBdsh1 = -T(SH_C1) * y;
            T dRGBdsh2 = T(SH_C1) * z;
            T dRGBdsh3 = -T(SH_C1) * x;
            dsh_ptr[1] = dRGBdsh1 * drgb;
            dsh_ptr[2] = dRGBdsh2 * drgb;
            dsh_ptr[3] = dRGBdsh3 * drgb;

            dRGBdx = -T(SH_C1) * sh[3];
            dRGBdy = -T(SH_C1) * sh[1];
            dRGBdz = T(SH_C1) * sh[2];

            if (Degree > 1)
            {{
                T xx = x * x, yy = y * y, zz = z * z;
                T xy = x * y, yz = y * z, xz = x * z;
                T dRGBdsh4 = T(SH_C2[0]) * xy;
                T dRGBdsh5 = T(SH_C2[1]) * yz;
                T dRGBdsh6 = T(SH_C2[2]) * (T(2) * zz - xx - yy);
                T dRGBdsh7 = T(SH_C2[3]) * xz;
                T dRGBdsh8 = T(SH_C2[4]) * (xx - yy);
                dsh_ptr[4] = dRGBdsh4 * drgb;
                dsh_ptr[5] = dRGBdsh5 * drgb;
                dsh_ptr[6] = dRGBdsh6 * drgb;
                dsh_ptr[7] = dRGBdsh7 * drgb;
                dsh_ptr[8] = dRGBdsh8 * drgb;

                dRGBdx += T(SH_C2[0]) * y * sh[4] + T(SH_C2[2]) * T(2) * -x * sh[6] + T(SH_C2[3]) * z * sh[7] + T(SH_C2[4]) * T(2) * x * sh[8];
                dRGBdy += T(SH_C2[0]) * x * sh[4] + T(SH_C2[1]) * z * sh[5] + T(SH_C2[2]) * T(2) * -y * sh[6] + T(SH_C2[4]) * T(2) * -y * sh[8];
                dRGBdz += T(SH_C2[1]) * y * sh[5] + T(SH_C2[2]) * T(2) * T(2) * z * sh[6] + T(SH_C2[3]) * x * sh[7];
                
                if (Degree > 2)
                {{
                    T dRGBdsh9 = T(SH_C3[0]) * y * (T(3) * xx - yy);
                    T dRGBdsh10 = T(SH_C3[1]) * xy * z;
                    T dRGBdsh11 = T(SH_C3[2]) * y * (T(4) * zz - xx - yy);
                    T dRGBdsh12 = T(SH_C3[3]) * z * (T(2) * zz - T(3) * xx - T(3) * yy);
                    T dRGBdsh13 = T(SH_C3[4]) * x * (T(4) * zz - xx - yy);
                    T dRGBdsh14 = T(SH_C3[5]) * z * (xx - yy);
                    T dRGBdsh15 = T(SH_C3[6]) * x * (xx - T(3) * yy);
                    dsh_ptr[9] = dRGBdsh9 * drgb;
                    dsh_ptr[10] = dRGBdsh10 * drgb;
                    dsh_ptr[11] = dRGBdsh11 * drgb;
                    dsh_ptr[12] = dRGBdsh12 * drgb;
                    dsh_ptr[13] = dRGBdsh13 * drgb;
                    dsh_ptr[14] = dRGBdsh14 * drgb;
                    dsh_ptr[15] = dRGBdsh15 * drgb;

                    dRGBdx += (
                        T(SH_C3[0]) * sh[9] * T(3) * T(2) * xy +
                        T(SH_C3[1]) * sh[10] * yz +
                        T(SH_C3[2]) * sh[11] * -T(2) * xy +
                        T(SH_C3[3]) * sh[12] * -T(3) * T(2) * xz +
                        T(SH_C3[4]) * sh[13] * (-T(3) * xx + T(4) * zz - yy) +
                        T(SH_C3[5]) * sh[14] * T(2) * xz +
                        T(SH_C3[6]) * sh[15] * T(3) * (xx - yy));

                    dRGBdy += (
                        T(SH_C3[0]) * sh[9] * T(3) * (xx - yy) +
                        T(SH_C3[1]) * sh[10] * xz +
                        T(SH_C3[2]) * sh[11] * (-T(3) * yy + T(4) * zz - xx) +
                        T(SH_C3[3]) * sh[12] * -T(3) * T(2) * yz +
                        T(SH_C3[4]) * sh[13] * -T(2) * xy +
                        T(SH_C3[5]) * sh[14] * -T(2) * yz +
                        T(SH_C3[6]) * sh[15] * -T(3) * T(2) * xy);

                    dRGBdz += (
                        T(SH_C3[1]) * sh[10] * xy +
                        T(SH_C3[2]) * sh[11] * T(4) * T(2) * yz +
                        T(SH_C3[3]) * sh[12] * T(3) * (T(2) * zz - xx - yy) +
                        T(SH_C3[4]) * sh[13] * T(4) * T(2) * xz +
                        T(SH_C3[5]) * sh[14] * (xx - yy));
                }}
            }}
        }}
        tv::array<T, 3> ddir{{dRGBdx.template op<op::dot>(drgb), dRGBdy.template op<op::dot>(drgb), dRGBdz.template op<op::dot>(drgb)}};
        return ddir;
        """)
        return code.ret("tv::array<T, 3>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def scale_quat_to_cov3d(self):
        """cov3d = R @ S @ S.T @ R.T
        """
        code = pccm.code()
        code.targ("T")
        code.arg("scale", "tv::array<T, 3>")
        code.arg("quat", "tv::array<T, 4>")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        // auto S = scale.template op<op::from_diagonal>();
        auto R = quat.template op<op::uqmat_colmajor>(); // .template op<op::transpose>();
        auto M = R * op::reshape<-1, 1>(scale);
        auto sigma = M.template op<op::mm_nnn>(M.template op<op::transpose>());
        return {{sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1], sigma[1][2], sigma[2][2]}};
        """)
        return code.ret("tv::array<T, 6>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def scale_quat_to_cov3d_grad(self):
        """cov3d = R @ S @ S.T @ R.T
        """
        code = pccm.code()
        code.targ("T")
        code.arg("dcov3d_vec", "tv::array<T, 6>")

        code.arg("scale", "tv::array<T, 3>")
        code.arg("quat", "tv::array<T, 4>")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto S = scale.template op<op::from_diagonal>();
        auto R = quat.template op<op::uqmat_colmajor>(); // .template op<op::transpose>();
        // auto M = R.template op<op::mm_nnn>(S);
        auto M = R * op::reshape<-1, 1>(scale);
        tv::array_nd<T, 3, 3> dL_dsigma{{
            tv::array<T, 3>{{dcov3d_vec[0], T(0.5) * dcov3d_vec[1], T(0.5) * dcov3d_vec[2]}},
            tv::array<T, 3>{{T(0.5) * dcov3d_vec[1], dcov3d_vec[3], T(0.5) * dcov3d_vec[4]}},
            tv::array<T, 3>{{T(0.5) * dcov3d_vec[2], T(0.5) * dcov3d_vec[4], dcov3d_vec[5]}}
        }};
        auto dL_dM_T = T(2) * dL_dsigma.template op<op::mm_nnn>(M);
        tv::array<T, 3> dL_dscale{{
            dL_dM_T[0].template op<op::dot>(R[0]),
            dL_dM_T[1].template op<op::dot>(R[1]),
            dL_dM_T[2].template op<op::dot>(R[2])
        }};
        // 
        auto dL_dquat = (dL_dM_T * op::reshape<-1, 1>(scale)).template op<op::uqmat_colmajor_grad>(quat);
        return std::make_tuple(dL_dscale, dL_dquat);
        """)
        return code.ret("std::tuple<tv::array<T, 3>, tv::array<T, 4>>")

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
        
        auto W_T_matmul_J_T_cm = world2cam_T_cm.template op<op::mm_nnn>(J_T_cm);
        tv::array_nd<T, 3, 3> Vrk{{
            tv::array<T, 3>{{cov3d_vec[0], cov3d_vec[1], cov3d_vec[2]}},
            tv::array<T, 3>{{cov3d_vec[1], cov3d_vec[3], cov3d_vec[4]}},
            tv::array<T, 3>{{cov3d_vec[2], cov3d_vec[4], cov3d_vec[5]}}
        }};
        // Vrk.T = Vrk
        // Projected Cov = J @ W @ Cov @ W.T @ J.T
        // = W_T_matmul_J_T_cm.T @ Vrk @ W_T_matmul_J_T_cm
        // auto cov_projected = W_T_matmul_J_T_cm.template op<op::transpose>().template op<op::mm_nnn>(Vrk).template op<op::mm_nnn>(W_T_matmul_J_T_cm);
        
        auto cov_projected = W_T_matmul_J_T_cm.template op<op::mm_tnn>(Vrk).template op<op::mm_nnn>(W_T_matmul_J_T_cm);
        return {{cov_projected[0][0], cov_projected[0][1], cov_projected[1][1]}};
        """)
        return code.ret("tv::array<T, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def project_gaussian_to_2d_grad(self):
        code = pccm.code()
        code.targ("T")
        code.arg("dcov2d", "tv::array<T, 3>")

        code.arg("mean_camera", "tv::array<T, 3>")
        code.arg("focal_length, tan_fov", "tv::array<T, 2>")
        code.arg("cam2world_T", "tv::array_nd<T, 4, 3>")
        code.arg("cov3d_vec", "tv::array_nd<T, 6>")

        code.arg("clamp_factor", "T", "1.3f")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto limit = tan_fov * clamp_factor * mean_camera[2];
        auto txylimit_focal = op::slice<0, 2>(mean_camera).template op<op::clamp>(-limit, limit) * focal_length;

        T x_limit_grad = (mean_camera[0] < -limit[0] || mean_camera[0] > limit[0]) ? T(0) : T(1);
        T y_limit_grad = (mean_camera[1] < -limit[1] || mean_camera[1] > limit[1]) ? T(0) : T(1);
        auto tz_square = mean_camera[2] * mean_camera[2];
        auto tz_square_inv = T(1) / tz_square;
        auto tz_3 = mean_camera[2] * mean_camera[2] * mean_camera[2];
        tv::array_nd<T, 2, 2> dcov_2d{{
            tv::array<T, 2>{{dcov2d[0], T(0.5) * dcov2d[1]}},
            tv::array<T, 2>{{T(0.5) * dcov2d[1], dcov2d[2]}},
        }};
        tv::array_nd<T, 2, 3> J_T_cm{{
            tv::array<T, 3>{{focal_length[0] / mean_camera[2], 0, -txylimit_focal[0] * tz_square_inv}},
            tv::array<T, 3>{{0, focal_length[1] / mean_camera[2], -txylimit_focal[1] * tz_square_inv}},
        }};
        auto world2cam_T_cm = op::slice<0, 3>(cam2world_T);
        
        tv::array_nd<T, 2, 3> W_T_matmul_J_T_cm = world2cam_T_cm.template op<op::mm_nnn>(J_T_cm);
        tv::array_nd<T, 3, 3> Vrk{{
            tv::array<T, 3>{{cov3d_vec[0], cov3d_vec[1], cov3d_vec[2]}},
            tv::array<T, 3>{{cov3d_vec[1], cov3d_vec[3], cov3d_vec[4]}},
            tv::array<T, 3>{{cov3d_vec[2], cov3d_vec[4], cov3d_vec[5]}}
        }};
        // Vrk.T = Vrk
        // Projected Cov = J @ W @ Cov @ W.T @ J.T
        // = W_T_matmul_J_T_cm.T @ Vrk @ W_T_matmul_J_T_cm
        auto dVrk = dcov_2d.template op<op::variance_transform_nnn_grad_rfs>(W_T_matmul_J_T_cm.template op<op::transpose>(), Vrk);
        
        tv::array<T, 6> dcov3d_vec{{
            dVrk[0][0], T(2) * dVrk[0][1], T(2) * dVrk[0][2],
            dVrk[1][1], T(2) * dVrk[1][2], dVrk[2][2]
        }};
        tv::array_nd<T, 2, 3> dW_T_matmul_J_T_cm = dcov_2d.template op<op::variance_transform_nnn_grad_lfs>(W_T_matmul_J_T_cm.template op<op::transpose>(), Vrk).template op<op::transpose>();
        auto dJ_T_cm = dW_T_matmul_J_T_cm.template op<op::mm_nnn_grad_rfs>(world2cam_T_cm, J_T_cm);
        tv::array<T, 3> dmean{{
            -dJ_T_cm[0][2] * focal_length[0] * tz_square_inv * x_limit_grad,
            -dJ_T_cm[1][2] * focal_length[1] * tz_square_inv * y_limit_grad,
            (-(focal_length[0] * dJ_T_cm[0][0] + focal_length[1] * dJ_T_cm[1][1]) * tz_square_inv + 
             2 * (dJ_T_cm[0][2] * txylimit_focal[0] + dJ_T_cm[1][2] * txylimit_focal[1]) / tz_3)
        }};
        return std::make_tuple(dmean, dcov3d_vec);
        """)
        return code.ret("std::tuple<tv::array<T, 3>, tv::array_nd<T, 6>>")


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

        T mid = T(0.5) * (cov2d_vec[0] + cov2d_vec[2]);
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

