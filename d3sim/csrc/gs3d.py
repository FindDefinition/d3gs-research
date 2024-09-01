import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC
from d3sim.core.geodef import EulerIntrinsicOrder

from d3sim.core import dataclass_dispatch as dataclasses
from ccimport import compat
from cumm.inliner.sympy_codegen import sigmoid, tvarray_math_expr, VectorSymOperator, Scalar, Vector, VectorExpr
import sympy


class Cov2dInvWithCompOp(VectorSymOperator):
    def __init__(self, lowpass_filter: float = 0.3):
        super().__init__()
        self.lowpass_filter = lowpass_filter

    def forward(self, a: sympy.Symbol, 
                b: sympy.Symbol,
                c: sympy.Symbol,
                lf: sympy.Symbol) -> dict[str, VectorExpr | sympy.Expr]:
        det_l = a * c - b * b
        a_l = a - lf
        c_l = c - lf
        det = a_l * c_l - b * b
        comp = det / det_l
        det_inv = 1.0 / det_l

        
        return {
            "a_inv": c * det_inv,
            "b_inv": -b * det_inv,
            "c_inv": a * det_inv,
            "comp": comp
        }

class GaussianEllipseOp(VectorSymOperator):
    def forward(self, x, y, u, v, cx, cy, cz, bound) -> dict[str, VectorExpr | sympy.Expr]:
        """ellipse: -conic_opacity_vec[0] * (center[0] - x[0]) ^ 2 + 
                    -conic_opacity_vec[2] * (center[1] - x[1]) ^ 2 -
                    2 * conic_opacity_vec[1] * (center[0] - x[0]) * (center[1] - x[1])
                    = 2 * log(1 / (255 * conic_opacity_vec[3]))
        calc interacts with aabb
        """

        res = -cx * (u - x) ** 2 - cz * (v - y) ** 2 - 2 * cy * (u - x) * (v - y) - bound
        return {
            "res": res
        }

class GaussianEllipseCenterOp(VectorSymOperator):
    def forward(self, x, y, cx, cy, cz, bound) -> dict[str, VectorExpr | sympy.Expr]:
        """ellipse: -conic_opacity_vec[0] * (center[0] - x[0]) ^ 2 + 
                    -conic_opacity_vec[2] * (center[1] - x[1]) ^ 2 -
                    2 * conic_opacity_vec[1] * (center[0] - x[0]) * (center[1] - x[1])
                    = 2 * log(1 / (255 * conic_opacity_vec[3]))
        calc interacts with aabb
        """

        res = -cx * ( - x) ** 2 - cz * ( - y) ** 2 - 2 * cy * ( - x) * ( - y) - bound
        return {
            "res": res
        }



class EllipseBoundOp(VectorSymOperator):
    def forward(self, x, y, u, v, cx, cy, cz, bound) -> dict[str, VectorExpr | sympy.Expr]:
        # copy from sqrt part in GaussianEllipseOp
        res_x = -bound * cz - cx * cz * u * u + 2 * cx * cz * u * x - cx * cz * x * x + cy * cy * u * u - 2 * cy * cy * u * x + cy * cy * x * x
        res_y = -bound * cx - cx * cz * v * v + 2 * cx * cz * v * y - cx * cz * y * y + cy * cy * v * v - 2 * cy * cy * v * y + cy * cy * y * y
        return {
            "res_x": res_x,
            "res_y": res_y
        }

_COV2D_INV_OP = Cov2dInvWithCompOp().build()

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
        # if compat.IsAppleSiliconMacOs:
        #     self.add_code_before_class("""
        #     constant float SH_C0 = 0.28209479177387814f;
        #     constant float SH_C1 = 0.4886025119029199f;
        #     constant float SH_C2[5] = {
        #         1.0925484305920792f,
        #         -1.0925484305920792f,
        #         0.31539156525252005f,
        #         -1.0925484305920792f,
        #         0.5462742152960396f
        #     };
        #     constant float SH_C3[7] = {
        #         -0.5900435899266435f,
        #         2.890611442640554f,
        #         -0.4570457994644658f,
        #         0.3731763325901154f,
        #         -0.4570457994644658f,
        #         1.445305721320277f,
        #         -0.5900435899266435f
        #     };
        #     """)
        # else:
        #     self.add_code_before_class("""
        #     __device__ const float SH_C0 = 0.28209479177387814f;
        #     __device__ const float SH_C1 = 0.4886025119029199f;
        #     __device__ const float SH_C2[5] = {
        #         1.0925484305920792f,
        #         -1.0925484305920792f,
        #         0.31539156525252005f,
        #         -1.0925484305920792f,
        #         0.5462742152960396f
        #     };
        #     __device__ const float SH_C3[7] = {
        #         -0.5900435899266435f,
        #         2.890611442640554f,
        #         -0.4570457994644658f,
        #         0.3731763325901154f,
        #         -0.4570457994644658f,
        #         1.445305721320277f,
        #         -0.5900435899266435f
        #     };
        #     """)
    
    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def sh_dir_to_rgb(self):
        code = pccm.code()
        code.nontype_targ("Degree", "int")
        code.targ("T")
        code.arg("dir", "tv::array<T, 3>")
        code.arg("sh_ptr", "TV_METAL_DEVICE const tv::array<T, 3>*")
        code.arg("sh_base_ptr", "TV_METAL_DEVICE const tv::array<T, 3>*", "nullptr")

        code.raw(f"""
        namespace op = tv::arrayops;
        op::PointerValueReader<TV_METAL_DEVICE const tv::array<T, 3>> sh(sh_ptr + int(sh_base_ptr == nullptr));
        static_assert(Degree >= 0 && Degree <= 3, "Degree must be in [0, 3]");
        auto base = sh_base_ptr == nullptr ? sh_ptr[0] : sh_base_ptr[0];
        tv::array<T, 3> result = T({SHConstants.C0}) * base;
        if (Degree > 0)
        {{
            T x = dir[0];
            T y = dir[1];
            T z = dir[2];
            constexpr T C1 = T({SHConstants.C1});
            result = result - C1 * y * sh[0] + C1 * z * sh[1] - C1 * x * sh[2];

            if (Degree > 1)
            {{
                T xx = x * x, yy = y * y, zz = z * z;
                T xy = x * y, yz = y * z, xz = x * z;
                result = result +
                    T({SHConstants.C2[0]}) * xy * sh[3] +
                    T({SHConstants.C2[1]}) * yz * sh[4] +
                    T({SHConstants.C2[2]}) * (T(2) * zz - xx - yy) * sh[5] +
                    T({SHConstants.C2[3]}) * xz * sh[6] +
                    T({SHConstants.C2[4]}) * (xx - yy) * sh[7];

                if (Degree > 2)
                {{
                    result = result +
                        T({SHConstants.C3[0]}) * y * (T(3) * xx - yy) * sh[8] +
                        T({SHConstants.C3[1]}) * xy * z * sh[9] +
                        T({SHConstants.C3[2]}) * y * (T(4) * zz - xx - yy) * sh[10] +
                        T({SHConstants.C3[3]}) * z * (T(2) * zz - T(3) * xx - T(3) * yy) * sh[11] +
                        T({SHConstants.C3[4]}) * x * (T(4) * zz - xx - yy) * sh[12] +
                        T({SHConstants.C3[5]}) * z * (xx - yy) * sh[13] +
                        T({SHConstants.C3[6]}) * x * (xx - T(3) * yy) * sh[14];
                }}
            }}
        }}
        result += T(0.5);
        return result;
        """)
        return code.ret("tv::array<T, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def sh_dir_to_rgb_grad(self):
        return self.sh_dir_to_rgb_grad_template(False)

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def sh_dir_to_rgb_grad_batch(self):
        return self.sh_dir_to_rgb_grad_template(True)

    def sh_dir_to_rgb_grad_template(self, is_batch: bool):
        code = pccm.code()
        assign_or_add = "+=" if is_batch else "="
        code.nontype_targ("Degree", "int")
        code.targ("T")
        code.arg("drgb", "tv::array<T, 3>")
        code.arg("dsh_ptr", "TV_METAL_DEVICE tv::array<T, 3>*")
        code.arg("dir", "tv::array<T, 3>")
        code.arg("sh_ptr", "TV_METAL_DEVICE const tv::array<T, 3>*")
        code.arg("dsh_base_ptr", "TV_METAL_DEVICE tv::array<T, 3>*", "nullptr")

        code.raw(f"""
        namespace op = tv::arrayops;
        op::PointerValueReader<TV_METAL_DEVICE const tv::array<T, 3>> sh(sh_ptr + int(dsh_base_ptr == nullptr));
        static_assert(Degree >= 0 && Degree <= 3, "Degree must be in [0, 3]");
        tv::array<T, 3> dRGBdx{{}};
        tv::array<T, 3> dRGBdy{{}};
        tv::array<T, 3> dRGBdz{{}};
        auto real_dsh_base_ptr = dsh_base_ptr == nullptr ? dsh_ptr : dsh_base_ptr;
        dsh_ptr += int(dsh_base_ptr == nullptr);
        T x = dir[0];
        T y = dir[1];
        T z = dir[2];

        real_dsh_base_ptr[0] {assign_or_add} T({SHConstants.C0}) * drgb;
        if (Degree > 0)
        {{
            T dRGBdsh1 = -T({SHConstants.C1}) * y;
            T dRGBdsh2 = T({SHConstants.C1}) * z;
            T dRGBdsh3 = -T({SHConstants.C1}) * x;
            dsh_ptr[0] {assign_or_add} dRGBdsh1 * drgb;
            dsh_ptr[1] {assign_or_add} dRGBdsh2 * drgb;
            dsh_ptr[2] {assign_or_add} dRGBdsh3 * drgb;

            dRGBdx = -T({SHConstants.C1}) * sh[2];
            dRGBdy = -T({SHConstants.C1}) * sh[0];
            dRGBdz = T({SHConstants.C1}) * sh[1];

            if (Degree > 1)
            {{
                T xx = x * x, yy = y * y, zz = z * z;
                T xy = x * y, yz = y * z, xz = x * z;
                T dRGBdsh4 = T({SHConstants.C2[0]}) * xy;
                T dRGBdsh5 = T({SHConstants.C2[1]}) * yz;
                T dRGBdsh6 = T({SHConstants.C2[2]}) * (T(2) * zz - xx - yy);
                T dRGBdsh7 = T({SHConstants.C2[3]}) * xz;
                T dRGBdsh8 = T({SHConstants.C2[4]}) * (xx - yy);
                dsh_ptr[3] {assign_or_add} dRGBdsh4 * drgb;
                dsh_ptr[4] {assign_or_add} dRGBdsh5 * drgb;
                dsh_ptr[5] {assign_or_add} dRGBdsh6 * drgb;
                dsh_ptr[6] {assign_or_add} dRGBdsh7 * drgb;
                dsh_ptr[7] {assign_or_add} dRGBdsh8 * drgb;
                
                dRGBdx += T({SHConstants.C2[0]}) * y * sh[3] + T({SHConstants.C2[2]}) * T(2) * -x * sh[5] + T({SHConstants.C2[3]}) * z * sh[6] + T({SHConstants.C2[4]}) * T(2) * x * sh[7];
                dRGBdy += T({SHConstants.C2[0]}) * x * sh[3] + T({SHConstants.C2[1]}) * z * sh[4] + T({SHConstants.C2[2]}) * T(2) * -y * sh[5] + T({SHConstants.C2[4]}) * T(2) * -y * sh[7];
                dRGBdz += T({SHConstants.C2[1]}) * y * sh[4] + T({SHConstants.C2[2]}) * T(2) * T(2) * z * sh[5] + T({SHConstants.C2[3]}) * x * sh[6];
                
                if (Degree > 2)
                {{
                    T dRGBdsh9 = T({SHConstants.C3[0]}) * y * (T(3) * xx - yy);
                    T dRGBdsh10 = T({SHConstants.C3[1]}) * xy * z;
                    T dRGBdsh11 = T({SHConstants.C3[2]}) * y * (T(4) * zz - xx - yy);
                    T dRGBdsh12 = T({SHConstants.C3[3]}) * z * (T(2) * zz - T(3) * xx - T(3) * yy);
                    T dRGBdsh13 = T({SHConstants.C3[4]}) * x * (T(4) * zz - xx - yy);
                    T dRGBdsh14 = T({SHConstants.C3[5]}) * z * (xx - yy);
                    T dRGBdsh15 = T({SHConstants.C3[6]}) * x * (xx - T(3) * yy);
                    dsh_ptr[8] {assign_or_add} dRGBdsh9 * drgb;
                    dsh_ptr[9] {assign_or_add} dRGBdsh10 * drgb;
                    dsh_ptr[10] {assign_or_add} dRGBdsh11 * drgb;
                    dsh_ptr[11] {assign_or_add} dRGBdsh12 * drgb;
                    dsh_ptr[12] {assign_or_add} dRGBdsh13 * drgb;
                    dsh_ptr[13] {assign_or_add} dRGBdsh14 * drgb;
                    dsh_ptr[14] {assign_or_add} dRGBdsh15 * drgb;

                    dRGBdx += (
                        T({SHConstants.C3[0]}) * sh[8] * T(3) * T(2) * xy +
                        T({SHConstants.C3[1]}) * sh[9] * yz +
                        T({SHConstants.C3[2]}) * sh[10] * -T(2) * xy +
                        T({SHConstants.C3[3]}) * sh[11] * -T(3) * T(2) * xz +
                        T({SHConstants.C3[4]}) * sh[12] * (-T(3) * xx + T(4) * zz - yy) +
                        T({SHConstants.C3[5]}) * sh[13] * T(2) * xz +
                        T({SHConstants.C3[6]}) * sh[14] * T(3) * (xx - yy));

                    dRGBdy += (
                        T({SHConstants.C3[0]}) * sh[8] * T(3) * (xx - yy) +
                        T({SHConstants.C3[1]}) * sh[9] * xz +
                        T({SHConstants.C3[2]}) * sh[10] * (-T(3) * yy + T(4) * zz - xx) +
                        T({SHConstants.C3[3]}) * sh[11] * -T(3) * T(2) * yz +
                        T({SHConstants.C3[4]}) * sh[12] * -T(2) * xy +
                        T({SHConstants.C3[5]}) * sh[13] * -T(2) * yz +
                        T({SHConstants.C3[6]}) * sh[14] * -T(3) * T(2) * xy);

                    dRGBdz += (
                        T({SHConstants.C3[1]}) * sh[9] * xy +
                        T({SHConstants.C3[2]}) * sh[10] * T(4) * T(2) * yz +
                        T({SHConstants.C3[3]}) * sh[11] * T(3) * (T(2) * zz - xx - yy) +
                        T({SHConstants.C3[4]}) * sh[12] * T(4) * T(2) * xz +
                        T({SHConstants.C3[5]}) * sh[13] * (xx - yy));
                }}
            }}
        }}
        tv::array<T, 3> ddir{{dRGBdx.template op<op::dot>(drgb), dRGBdy.template op<op::dot>(drgb), dRGBdz.template op<op::dot>(drgb)}};
        return ddir;
        """)
        return code.ret("tv::array<T, 3>")

    # @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def sh_dir_to_rgb_grad_v2(self):
        code = pccm.code()
        code.nontype_targ("Degree", "int")
        code.arg("drgb", "tv::array<float, 3>")
        code.arg("dsh_ptr", "TV_METAL_DEVICE tv::array<float, 3>*")
        code.arg("dir", "tv::array<float, 3>")
        code.arg("sh_ptr", "TV_METAL_DEVICE const tv::array<float, 3>*")
        code.raw(f"""
        namespace op = tv::arrayops;
        // op::PointerValueReader<TV_METAL_DEVICE const tv::array<float, 3>> sh(sh_ptr);
        auto sh = sh_ptr;
        static_assert(Degree >= 0 && Degree <= 3, "Degree must be in [0, 3]");
        tv::array<float, 3> dRGBdx{{}};
        tv::array<float, 3> dRGBdy{{}};
        tv::array<float, 3> dRGBdz{{}};

        float x = dir[0];
        float y = dir[1];
        float z = dir[2];

        dsh_ptr[0] = (SH_C0) * drgb;
        if (Degree > 0)
        {{
            float dRGBdsh1 = -(SH_C1) * y;
            float dRGBdsh2 = (SH_C1) * z;
            float dRGBdsh3 = -(SH_C1) * x;
            dsh_ptr[1] = dRGBdsh1 * drgb;
            dsh_ptr[2] = dRGBdsh2 * drgb;
            dsh_ptr[3] = dRGBdsh3 * drgb;

            dRGBdx = -(SH_C1) * sh[3];
            dRGBdy = -(SH_C1) * sh[1];
            dRGBdz = (SH_C1) * sh[2];

            if (Degree > 1)
            {{
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                float dRGBdsh4 = (SH_C2[0]) * xy;
                float dRGBdsh5 = (SH_C2[1]) * yz;
                float dRGBdsh6 = (SH_C2[2]) * (2.0f * zz - xx - yy);
                float dRGBdsh7 = (SH_C2[3]) * xz;
                float dRGBdsh8 = (SH_C2[4]) * (xx - yy);
                dsh_ptr[4] = dRGBdsh4 * drgb;
                dsh_ptr[5] = dRGBdsh5 * drgb;
                dsh_ptr[6] = dRGBdsh6 * drgb;
                dsh_ptr[7] = dRGBdsh7 * drgb;
                dsh_ptr[8] = dRGBdsh8 * drgb;

                dRGBdx += (SH_C2[0]) * y * sh[4] + (SH_C2[2]) * 2.0f * -x * sh[6] + (SH_C2[3]) * z * sh[7] + (SH_C2[4]) * 2.0f * x * sh[8];
                dRGBdy += (SH_C2[0]) * x * sh[4] + (SH_C2[1]) * z * sh[5] + (SH_C2[2]) * 2.0f * -y * sh[6] + (SH_C2[4]) * 2.0f * -y * sh[8];
                dRGBdz += (SH_C2[1]) * y * sh[5] + (SH_C2[2]) * 2.0f * 2.0f * z * sh[6] + (SH_C2[3]) * x * sh[7];
                
                if (Degree > 2)
                {{
                    float dRGBdsh9 = (SH_C3[0]) * y * (3.0f * xx - yy);
                    float dRGBdsh10 = (SH_C3[1]) * xy * z;
                    float dRGBdsh11 = (SH_C3[2]) * y * (4.0f * zz - xx - yy);
                    float dRGBdsh12 = (SH_C3[3]) * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
                    float dRGBdsh13 = (SH_C3[4]) * x * (4.0f * zz - xx - yy);
                    float dRGBdsh14 = (SH_C3[5]) * z * (xx - yy);
                    float dRGBdsh15 = (SH_C3[6]) * x * (xx - 3.0f * yy);
                    dsh_ptr[9] = dRGBdsh9 * drgb;
                    dsh_ptr[10] = dRGBdsh10 * drgb;
                    dsh_ptr[11] = dRGBdsh11 * drgb;
                    dsh_ptr[12] = dRGBdsh12 * drgb;
                    dsh_ptr[13] = dRGBdsh13 * drgb;
                    dsh_ptr[14] = dRGBdsh14 * drgb;
                    dsh_ptr[15] = dRGBdsh15 * drgb;

                    dRGBdx += (
                        (SH_C3[0]) * sh[9] * 3.0f * 2.0f * xy +
                        (SH_C3[1]) * sh[10] * yz +
                        (SH_C3[2]) * sh[11] * -2.0f * xy +
                        (SH_C3[3]) * sh[12] * -3.0f * 2.0f * xz +
                        (SH_C3[4]) * sh[13] * (-3.0f * xx + 4.0f * zz - yy) +
                        (SH_C3[5]) * sh[14] * 2.0f * xz +
                        (SH_C3[6]) * sh[15] * 3.0f * (xx - yy));

                    dRGBdy += (
                        (SH_C3[0]) * sh[9] * 3.0f * (xx - yy) +
                        (SH_C3[1]) * sh[10] * xz +
                        (SH_C3[2]) * sh[11] * (-3.0f * yy + 4.0f * zz - xx) +
                        (SH_C3[3]) * sh[12] * -3.0f * 2.0f * yz +
                        (SH_C3[4]) * sh[13] * -2.0f * xy +
                        (SH_C3[5]) * sh[14] * -2.0f * yz +
                        (SH_C3[6]) * sh[15] * -3.0f * 2.0f * xy);

                    dRGBdz += (
                        (SH_C3[1]) * sh[10] * xy +
                        (SH_C3[2]) * sh[11] * 4.0f * 2.0f * yz +
                        (SH_C3[3]) * sh[12] * 3.0f * (2.0f * zz - xx - yy) +
                        (SH_C3[4]) * sh[13] * 4.0f * 2.0f * xz +
                        (SH_C3[5]) * sh[14] * (xx - yy));
                }}
            }}
        }}
        tv::array<float, 3> ddir{{dRGBdx.template op<op::dot>(drgb), dRGBdy.template op<op::dot>(drgb), dRGBdz.template op<op::dot>(drgb)}};
        return ddir;
        """)
        return code.ret("tv::array<float, 3>")

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
        auto dL_dquat = (dL_dM_T * op::reshape<-1, 1>(scale)).template op<op::uqmat_colmajor_grad>(quat);
        return std::make_tuple(dL_dscale, dL_dquat);
        """)
        return code.ret("std::tuple<tv::array<T, 3>, tv::array<T, 4>>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def project_gaussian_to_2d(self):
        code = pccm.code()
        code.targ("T")
        code.nontype_targ("C2wRows", "size_t")

        code.arg("mean_camera", "tv::array<T, 3>")
        code.arg("focal_length, tan_fov", "tv::array<T, 2>")
        code.arg("cam2world_T", "tv::array_nd<T, C2wRows, 3>")
        code.arg("cov3d_vec", "tv::array_nd<T, 6>")

        code.arg("clamp_factor", "T", "1.3f")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto limit = (tan_fov * clamp_factor * mean_camera[2]).template op<op::abs>();
        auto txylimit = op::slice<0, 2>(mean_camera).template op<op::clamp>(-limit, limit);
	    // txylimit[0] = min(limit[0], max(-limit[0], mean_camera[0]));
	    // txylimit[1] = min(limit[1], max(-limit[1], mean_camera[1]));

        auto txylimit_focal = txylimit * focal_length;
        auto tz_square = mean_camera[2] * mean_camera[2];
        // we need T = W.T @ J.T, if @ is colmajor, we need 
        // W.T stored as colmajor and J.T stored as colmajor,
        // so store W as rowmajor and J as rowmajor.
        // J_T_cm = J_rm
        tv::array_nd<T, 2, 3> J_T_cm{{
            tv::array<T, 3>{{focal_length[0] / mean_camera[2], 0, -txylimit_focal[0] / tz_square}},
            tv::array<T, 3>{{0, focal_length[1] / mean_camera[2], -txylimit_focal[1] / tz_square}},
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
        code.nontype_targ("C2wRows", "size_t")

        code.arg("dcov2d", "tv::array<T, 3>")

        code.arg("mean_camera", "tv::array<T, 3>")
        code.arg("focal_length, tan_fov", "tv::array<T, 2>")
        code.arg("cam2world_T", "tv::array_nd<T, C2wRows, 3>")
        code.arg("cov3d_vec", "tv::array_nd<T, 6>")

        code.arg("clamp_factor", "T", "1.3f")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto limit = (tan_fov * clamp_factor * mean_camera[2]).template op<op::abs>();
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
        tv::array_nd<T, 2, 3> dW_T_matmul_J_T_cm = dcov_2d.template op<op::symmetric_variance_transform_nnn_grad_lfs>(W_T_matmul_J_T_cm.template op<op::transpose>(), Vrk).template op<op::transpose>();
        tv::array_nd<T, 2, 3> dJ_T_cm = dW_T_matmul_J_T_cm.template op<op::mm_nnn_grad_rfs>(world2cam_T_cm, J_T_cm);
        tv::array<T, 3> dmean{{
            x_limit_grad * -dJ_T_cm[0][2] * focal_length[0] * tz_square_inv,
            y_limit_grad * -dJ_T_cm[1][2] * focal_length[1] * tz_square_inv,
            (-(focal_length[0] * dJ_T_cm[0][0] + focal_length[1] * dJ_T_cm[1][1]) * tz_square_inv + 
             T(2) * (dJ_T_cm[0][2] * txylimit_focal[0] + dJ_T_cm[1][2] * txylimit_focal[1]) / tz_3)
        }};
        return std::make_tuple(dmean, dcov3d_vec);
        """)
        return code.ret("std::tuple<tv::array<T, 3>, tv::array_nd<T, 6>>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gaussian_2d_inverse_and_det(self):
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_vec", "tv::array<T, 3>")
        code.arg("lowpass_filter", "T")
        code.arg("eps", "T", "1e-6")
        code.raw(f"""
        namespace op = tv::arrayops;
        auto a = cov2d_vec[0];
        auto b = cov2d_vec[1];
        auto c = cov2d_vec[2];
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        T det = a * c - b * b;
        T det_inv = T(1) / (det + eps);
        tv::array<T, 3> ret{{
            c * det_inv,
            -b * det_inv,
            a * det_inv,
        }};
        return std::make_tuple(ret, det);
        """)
        return code.ret("std::tuple<tv::array<T, 3>, T>")
    
    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gaussian_2d_inverse_and_det_grad(self):
        code = pccm.code()
        code.targ("T")
        code.arg("dcov2d_inv_vec", "tv::array<T, 3>")

        code.arg("cov2d_vec", "tv::array<T, 3>")
        code.arg("lowpass_filter", "T")
        code.arg("eps", "T", "1e-8")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        T a = cov2d_vec[0];
        T b = cov2d_vec[1];
        T c = cov2d_vec[2];
        T det = a * c - b * b;
        T det2_inv = T(1) / (det * det + eps);
        // in original code, the dcov2d_inv_vec[1] is multipled by 2.
        // this is due to original code treat dcov2d_inv_vec as 2x2 symmetric matrix,
        // dcov2d_inv_vec is [da, db; db, dc]
        // so dcov2d_inv_vec[2] need to be multiplied by 2.
        // in our implementation, we treat dcov2d_inv_vec as a regular vector.
        tv::array<T, 3> ret{{
            det2_inv * (-c * c * dcov2d_inv_vec[0] + b * c * dcov2d_inv_vec[1] + (det - a * c) * dcov2d_inv_vec[2]),
            det2_inv * T(2) * (b * c * dcov2d_inv_vec[0] - (det / T(2) + b * b) * dcov2d_inv_vec[1] + a * b * dcov2d_inv_vec[2]),
            det2_inv * (-a * a * dcov2d_inv_vec[2] + a * b * dcov2d_inv_vec[1] + (det - a * c) * dcov2d_inv_vec[0]),
        }};
        return ret;
        """)
        return code.ret("tv::array<T, 3>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gaussian_2d_inverse_and_det_with_comp(self):
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_vec", "tv::array<T, 3>")
        code.arg("lowpass_filter", "T")

        code.arg("eps", "T", "1e-8")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto a = cov2d_vec[0] - lowpass_filter;
        auto c = cov2d_vec[2] - lowpass_filter;
        auto a_l = cov2d_vec[0];
        auto b = cov2d_vec[1];
        auto c_l = cov2d_vec[2];
        T det_lowpass = a_l * c_l - b * b;
        T det = a * c - b * b;
        T det_lowpass_inv = T(1) / det_lowpass;
        tv::array<T, 3> ret{{
            c_l * det_lowpass_inv,
            -b * det_lowpass_inv,
            a_l * det_lowpass_inv,
        }};
        return std::make_tuple(ret, det, det / det_lowpass);
        """)
        return code.ret("std::tuple<tv::array<T, 3>, T, T>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gaussian_2d_inverse_and_det_grad_with_comp(self):
        code = pccm.code()
        code.targ("T")
        code.arg("dcov2d_inv_vec", "tv::array<T, 3>")
        code.arg("ddet_div_det_lowpass", "T")

        code.arg("cov2d_vec", "tv::array<T, 3>")
        code.arg("lowpass_filter", "T")
        code.arg("eps", "T", "1e-8")
        use_sympy = False
        if use_sympy:
            # only for validation
            gdict = {
                "a_inv": "dcov2d_inv_vec[0]",
                "b_inv": "dcov2d_inv_vec[1]",
                "c_inv": "dcov2d_inv_vec[2]",
                "comp": "ddet_div_det_lowpass"
            }
            code.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<T>;
            T a = cov2d_vec[0];
            T b = cov2d_vec[1];
            T c = cov2d_vec[2];
            T lf = lowpass_filter;
            auto da = {_COV2D_INV_OP.generate_gradients_code("a", gdict)};
            auto db = {_COV2D_INV_OP.generate_gradients_code("b", gdict)};
            auto dc = {_COV2D_INV_OP.generate_gradients_code("c", gdict)};
            return tv::array<T, 3>{{da, db, dc}};
            """) 
            return code.ret("tv::array<T, 3>")
        else:
            code.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<T>;
            T a = cov2d_vec[0];
            T b = cov2d_vec[1];
            T c = cov2d_vec[2];
            T det = a * c - b * b;

            T det2_inv = T(1) / (det * det + eps);
            // in original code, the dcov2d_inv_vec[1] is multipled by 2.
            // this is due to original code treat dcov2d_inv_vec as 2x2 symmetric matrix,
            // dcov2d_inv_vec is [da, db; db, dc]
            // so dcov2d_inv_vec[2] need to be multiplied by 2.
            // in our implementation, we treat dcov2d_inv_vec as a regular vector.
            tv::array<T, 3> ret{{
                det2_inv * (-c * c * dcov2d_inv_vec[0] + b * c * dcov2d_inv_vec[1] + (det - a * c) * dcov2d_inv_vec[2]),
                det2_inv * T(2) * (b * c * dcov2d_inv_vec[0] - (det / T(2) + b * b) * dcov2d_inv_vec[1] + a * b * dcov2d_inv_vec[2]),
                det2_inv * (-a * a * dcov2d_inv_vec[2] + a * b * dcov2d_inv_vec[1] + (det - a * c) * dcov2d_inv_vec[0]),
            }};
            const T lf = lowpass_filter;
            const T grad_mul_det2_inv = ddet_div_det_lowpass * det2_inv;
            ret += tv::array<T, 3>{{
                lf * (c * c + b * b - c * lf) * grad_mul_det2_inv,
                T(-2) * b * lf * (a + c - lf) * grad_mul_det2_inv,
                lf * (a * a + b * b - a * lf) * grad_mul_det2_inv,
            }};
            return ret;
            """)
            return code.ret("tv::array<T, 3>")


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

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def get_gaussian_2d_ellipse_bound_aabb(self):
        """Get bounding aabb box of the ellipse.
        https://math.stackexchange.com/questions/1173868/show-that-a-2x2-matrix-a-is-symmetric-positive-definite-if-and-only-if-a-is-symm
        
        sympy result:
        ```
        {u - sqrt(-bound*cz*(cx*cz - cy**2))/(-cx*cz + cy**2), u + sqrt(-bound*cz*(cx*cz - cy**2))/(-cx*cz + cy**2)}

        {v - sqrt(-bound*cx*(cx*cz - cy**2))/(-cx*cz + cy**2), v + sqrt(-bound*cx*(cx*cz - cy**2))/(-cx*cz + cy**2)}
        ```
        """
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_inv_vec", "tv::array<T, 3>")
        code.arg("bound", "T")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        // from sympy, check __main in this file to get more info.

        T det = cov2d_inv_vec[0] * cov2d_inv_vec[2] - cov2d_inv_vec[1] * cov2d_inv_vec[1];
        T value_to_sqrt_x = -bound * cov2d_inv_vec[2] * det;
        T value_to_sqrt_y = -bound * cov2d_inv_vec[0] * det;
        T value_sqrted_x = math_op_t::sqrt(math_op_t::max(T(0), value_to_sqrt_x)) / det;
        T value_sqrted_y = math_op_t::sqrt(math_op_t::max(T(0), value_to_sqrt_y)) / det;
        value_sqrted_x = value_to_sqrt_x > 0 ? value_sqrted_x : 0;
        value_sqrted_y = value_to_sqrt_y > 0 ? value_sqrted_y : 0;

        return {{math_op_t::abs(value_sqrted_x), math_op_t::abs(value_sqrted_y)}};
        """)
        return code.ret("tv::array<T, 2>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def get_gaussian_2d_ellipse_width_height_vec(self):
        """ellipse: -conic_opacity_vec[0] * x ^ 2 + 
                    -conic_opacity_vec[2] * y ^ 2 -
                    2 * conic_opacity_vec[1] * x * y
                    = bound
        """

        # https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_inv_vec", "tv::array<T, 3>")
        code.arg("bound", "T")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        T ellipse_A = -cov2d_inv_vec[0] / bound;
        T ellipse_C = -cov2d_inv_vec[2] / bound;
        T ellipse_B_div_2 = -cov2d_inv_vec[1] / bound;
        // find eigenvalues
        T det = ellipse_A * ellipse_C - ellipse_B_div_2 * ellipse_B_div_2;

        T mid = T(0.5) * (ellipse_A + ellipse_C);

        T sqrt_part = math_op_t::sqrt(math_op_t::max(0.0f, mid * mid - det));
        T eigen1 = mid + sqrt_part;
        T eigen2 = mid - sqrt_part;

        T eigen1_sign = eigen1 > 0 ? 1 : -1;
        T eigen2_sign = eigen2 > 0 ? 1 : -1;
       
       // tv::array<T, 2> eigen_vec2{{eigen2 - ellipse_C, ellipse_B_div_2}};
        tv::array<T, 2> eigen_vec2{{eigen2 - ellipse_C, ellipse_B_div_2}};
        
        eigen_vec2 = eigen_vec2.template op<op::normalize>();
        // makesure eigen vec 1 is vec 2 rotate 90 degree
        tv::array<T, 2> eigen_vec1{{-eigen_vec2[1], eigen_vec2[0]}};

        // tv::array<T, 2> eigen_vec1{{eigen1 - ellipse_C, ellipse_B_div_2}};

        auto minor = math_op_t::rsqrt(eigen1 * eigen1_sign);
        auto major = math_op_t::rsqrt(eigen2 * eigen2_sign);
        return {{
            tv::array<T, 2>{{major, minor}},
            eigen_vec2.template op<op::normalize>() * major,
            eigen_vec1.template op<op::normalize>() * minor,
        }};
        """)
        return code.ret("tv::array_nd<T, 3, 2>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def get_gaussian_2d_ellipse_bound_rect_corners_and_major_vec(self):
        """Get rotated bounding box corners of the ellipse.
        clockwise.
        """
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_inv_vec", "tv::array<T, 3>")
        code.arg("bound", "T")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto eigen_val_vec = get_gaussian_2d_ellipse_width_height_vec(cov2d_inv_vec, bound);
        auto major_vec = eigen_val_vec[1];
        auto minor_vec = eigen_val_vec[2];
        auto p_top_right = minor_vec + major_vec;
        auto p_bottom_right = -minor_vec + major_vec;
        auto p_bottom_left = -minor_vec - major_vec;
        auto p_top_left = minor_vec - major_vec;
        return std::make_tuple(tv::array_nd<T, 4, 2>{{
            p_top_right, p_bottom_right, p_bottom_left, p_top_left
        }}, major_vec);
        """)
        return code.ret("std::tuple<tv::array_nd<T, 4, 2>, tv::array<T, 2>>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def prepare_dfvt_corners(self):
        """Prepare corners in specific order for 
        dual fast voxel traversal (DFVT).
        """
        code = pccm.code()
        code.nontype_targ("TileX", "int")
        code.nontype_targ("TileY", "int")

        code.targ("T")
        code.arg("cov2d_inv_vec", "tv::array<T, 3>")
        code.arg("center", "tv::array<T, 2>")

        code.arg("bound", "T")

        code.raw(f"""
        namespace op = tv::arrayops;
        constexpr tv::array<float, 2> kTileSize{{TileX, TileY}};
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto eigen_val_vec = get_gaussian_2d_ellipse_width_height_vec(cov2d_inv_vec, bound);
        auto major_vec = eigen_val_vec[1];
        auto minor_vec = eigen_val_vec[2];
        auto p_top_right = minor_vec + major_vec;
        auto p_bottom_right = -minor_vec + major_vec;
        auto p_bottom_left = -minor_vec - major_vec;
        auto p_top_left = minor_vec - major_vec;
        if (major_vec[0] >= 0 && major_vec[1] >= 0){{
            return {{
                (p_bottom_left + center) / kTileSize,
                (p_top_left + center) / kTileSize,
                (p_bottom_right + center) / kTileSize,
                (p_top_right + center) / kTileSize
            }};
        }} else if (major_vec[0] < 0 && major_vec[1] >= 0){{
            return {{
                (p_top_left + center) / kTileSize,
                (p_top_right + center) / kTileSize,
                (p_bottom_left + center) / kTileSize,
                (p_bottom_right + center) / kTileSize
            }};
        }} else if (major_vec[0] < 0 && major_vec[1] < 0){{
            return {{
                (p_top_right + center) / kTileSize,
                (p_bottom_right + center) / kTileSize,
                (p_top_left + center) / kTileSize,
                (p_bottom_left + center) / kTileSize
            }};
        }} else{{
            return {{
                (p_bottom_right + center) / kTileSize,
                (p_bottom_left + center) / kTileSize,
                (p_top_right + center) / kTileSize,
                (p_top_left + center) / kTileSize
            }};

        }}
        """)
        return code.ret("tv::array_nd<T, 4, 2>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def prepare_dfvt_corners_clipped(self):
        """Prepare corners in specific order for 
        dual fast voxel traversal (DFVT).
        corner format: start, left, right, end

        res corner format (clip by y = 0):
        start_left, start_right, left, right, end
        """
        code = pccm.code()
        code.nontype_targ("TileX", "int")
        code.nontype_targ("TileY", "int")

        code.targ("T")
        code.arg("cov2d_inv_vec", "tv::array<T, 3>")
        code.arg("center", "tv::array<T, 2>")

        code.arg("bound", "T")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto corners = prepare_dfvt_corners<TileX, TileY>(cov2d_inv_vec, center, bound);
        // calc each ray and y = 0 intersection
        auto start_left_dir = (corners[1] - corners[0]).template op<op::normalize>();
        auto start_right_dir = (corners[2] - corners[0]).template op<op::normalize>();
        auto left_end_dir = (corners[3] - corners[1]).template op<op::normalize>();
        auto right_end_dir = (corners[3] - corners[2]).template op<op::normalize>();

        auto start_left_y_zero_t = -corners[0][1] / start_left_dir[1];
        auto start_right_y_zero_t = -corners[0][1] / start_right_dir[1];
        auto left_end_y_zero_t = -corners[1][1] / left_end_dir[1];
        auto right_end_y_zero_t = -corners[2][1] / right_end_dir[1];

        auto start_left_y_zero_x = corners[0][0] + start_left_dir[0] * start_left_y_zero_t;
        auto start_right_y_zero_x = corners[0][0] + start_right_dir[0] * start_right_y_zero_t;
        auto left_end_y_zero_x = corners[1][0] + left_end_dir[0] * left_end_y_zero_t;
        auto right_end_y_zero_x = corners[2][0] + right_end_dir[0] * right_end_y_zero_t;

        tv::array<T, 2> start_left_y_zero{{start_left_y_zero_x, 0}};
        tv::array<T, 2> start_right_y_zero{{start_right_y_zero_x, 0}};
        tv::array<T, 2> left_end_y_zero{{left_end_y_zero_x, 0}};
        tv::array<T, 2> right_end_y_zero{{right_end_y_zero_x, 0}};


        bool is_start_y_le_zero = corners[0][1] <= 0;
        bool is_left_y_le_zero = corners[1][1] <= 0;
        bool is_right_y_le_zero = corners[2][1] <= 0;
        auto res_start_left = is_start_y_le_zero ? (is_left_y_le_zero ? left_end_y_zero : start_left_y_zero) : corners[0];
        auto res_left = is_left_y_le_zero ? left_end_y_zero : corners[1];
        auto res_start_right = is_start_y_le_zero ? (is_right_y_le_zero ? right_end_y_zero : start_right_y_zero) : corners[0];
        auto res_right = is_right_y_le_zero ? right_end_y_zero : corners[2];
        auto res_end = corners[3];
        tv::array_nd<T, 5, 2> corner_res {{
            res_start_left, res_start_right, res_left, res_right, res_end
        }};
        tv::array_nd<T, 4, 2> ray_dir_res {{
            start_left_dir, start_right_dir, left_end_dir, right_end_dir
        }};
        return std::make_tuple(corner_res, ray_dir_res);
        """)
        return code.ret("std::tuple<tv::array_nd<T, 5, 2>, tv::array_nd<T, 4, 2>>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def get_gaussian_2d_ellipse_bound_rect_aabb(self):
        """Get rotated bounding box corners of the ellipse.
        clockwise.
        """
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_inv_vec", "tv::array<T, 3>")
        code.arg("bound", "T")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto corners = get_gaussian_2d_ellipse_bound_rect_corners(cov2d_inv_vec, bound);
        auto corners_T = corners.template op<op::transpose>();
        return {{
            corners_T[0].template op<op::reduce_min>(),
            corners_T[1].template op<op::reduce_min>(),
            corners_T[0].template op<op::reduce_max>(),
            corners_T[1].template op<op::reduce_max>()
        }};
        """)
        return code.ret("tv::array_nd<T, 4>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def get_gaussian_2d_ellipse_bound_obb(self):
        """Get rotated bounding box obb (w, h, sin, cos) of the ellipse.
        clockwise.
        """
        code = pccm.code()
        code.targ("T")
        code.arg("cov2d_inv_vec", "tv::array<T, 3>")
        code.arg("bound", "T")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto eigen_val_vec = get_gaussian_2d_ellipse_width_height_vec(cov2d_inv_vec, bound);
        auto major_vec = eigen_val_vec[1].template op<op::normalize>();
        auto sin_theta = major_vec[1];
        auto cos_theta = major_vec[0];
        return {{
            eigen_val_vec[0][0] * 2, eigen_val_vec[0][1] * 2, sin_theta, cos_theta
        }};
        """)
        return code.ret("tv::array<T, 4>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gs2d_get_alpha(self):
        code = pccm.code()
        code.targ("T")
        code.arg("x, center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("alpha_max", "T", "0.99")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        tv::array<T, 2> dist{{center[0] - x[0], center[1] - x[1]}};
        T power = (T(-0.5) * (conic_opacity_vec[0] * dist[0] * dist[0] + 
                                conic_opacity_vec[2] * dist[1] * dist[1]) - 
                                conic_opacity_vec[1] * dist[0] * dist[1]);
        T G = math_op_t::fast_exp(power); 
        T alpha = math_op_t::min(alpha_max, conic_opacity_vec[3] * G);
        // alpha >= 1/ 255. so G >= 1 / (255 * conic_opacity_vec[3]),
        // power >= log(alpha_eps / conic_opacity_vec[3])
        return {{alpha, power}};
        """)
        return code.ret("tv::array<T, 2>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gs2d_get_2xpower(self):
        code = pccm.code()
        code.targ("T")
        code.arg("x, center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        tv::array<T, 2> dist{{center[0] - x[0], center[1] - x[1]}};
        T power = (-(conic_opacity_vec[0] * dist[0] * dist[0] + 
                                conic_opacity_vec[2] * dist[1] * dist[1]) - 
                                T(2) * conic_opacity_vec[1] * dist[0] * dist[1]);
        return power;
        """)
        return code.ret("T")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gs2d_ellipse_aabb_is_overlap(self):
        """ellipse: -0.5 * conic_opacity_vec[0] * (center[0] - x[0]) ^ 2 + 
                    -0.5 * conic_opacity_vec[2] * (center[1] - x[1]) ^ 2 -
                    conic_opacity_vec[1] * (center[0] - x[0]) * (center[1] - x[1])
                    = bound * 0.5
        bound: 2 * log(alpha_eps / conic_opacity_vec[3])

        calc ellipse is overlap with aabb

        VERY SLOW. consider `get_gaussian_2d_ellipse_bound_aabb` and `aabb_has_overlap_area`.
        """
        code = pccm.code()
        code.targ("T")
        code.arg("center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("aabb_min, aabb_max", "tv::array<T, 2>")
        code.arg("bound", "T") # 2 * log(alpha_eps / conic_opacity_vec[3])
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;

        auto int_x0 = solve_ellipse_x_constant(aabb_min[0], center, conic_opacity_vec, bound);
        auto int_x1 = solve_ellipse_x_constant(aabb_max[0], center, conic_opacity_vec, bound);
        auto int_y0 = solve_ellipse_y_constant(aabb_min[1], center, conic_opacity_vec, bound);
        auto int_y1 = solve_ellipse_y_constant(aabb_max[1], center, conic_opacity_vec, bound);
        bool valid = (std::get<1>(int_x0) >= aabb_min[1] && std::get<1>(int_x0) <= aabb_max[1]) || 
                     (std::get<2>(int_x0) >= aabb_min[1] && std::get<2>(int_x0) <= aabb_max[1]);
        valid |= (std::get<1>(int_x1) >= aabb_min[1] && std::get<1>(int_x1) <= aabb_max[1]) ||
                 (std::get<2>(int_x1) >= aabb_min[1] && std::get<2>(int_x1) <= aabb_max[1]);
        valid |= (std::get<1>(int_y0) >= aabb_min[0] && std::get<1>(int_y0) <= aabb_max[0]) ||
                    (std::get<2>(int_y0) >= aabb_min[0] && std::get<2>(int_y0) <= aabb_max[0]);
        valid |= (std::get<1>(int_y1) >= aabb_min[0] && std::get<1>(int_y1) <= aabb_max[0]) ||
                    (std::get<2>(int_y1) >= aabb_min[0] && std::get<2>(int_y1) <= aabb_max[0]);
        
        bool ellipse_cover_whole_aabb = std::get<1>(int_x0) <= aabb_min[1] && std::get<2>(int_x0) >= aabb_max[1];
        ellipse_cover_whole_aabb &= std::get<1>(int_x1) <= aabb_min[1] && std::get<2>(int_x1) >= aabb_max[1];
        ellipse_cover_whole_aabb &= std::get<1>(int_y0) <= aabb_min[0] && std::get<2>(int_y0) >= aabb_max[0];
        ellipse_cover_whole_aabb &= std::get<1>(int_y1) <= aabb_min[0] && std::get<2>(int_y1) >= aabb_max[0];
        
        bool aabb_cover_ellipse_center = aabb_min[0] <= center[0] && aabb_max[0] >= center[0] && aabb_min[1] <= center[1] && aabb_max[1] >= center[1];
        return valid || ellipse_cover_whole_aabb || aabb_cover_ellipse_center;
        """)
        return code.ret("bool")
        
    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gs2d_ellipse_aabb_is_overlap_external_inter(self):
        """ellipse: -0.5 * conic_opacity_vec[0] * (center[0] - x[0]) ^ 2 + 
                    -0.5 * conic_opacity_vec[2] * (center[1] - x[1]) ^ 2 -
                    conic_opacity_vec[1] * (center[0] - x[0]) * (center[1] - x[1])
                    = bound * 0.5
        bound: 2 * log(alpha_eps / conic_opacity_vec[3])

        calc ellipse is overlap with aabb

        VERY SLOW. consider `get_gaussian_2d_ellipse_bound_aabb` and `aabb_has_overlap_area`.
        """
        code = pccm.code()
        code.targ("T")
        code.arg("int_x0, int_x1, int_y0, int_y1", "std::tuple<bool, T, T>")
        code.arg("center", "tv::array<T, 2>")

        code.arg("aabb_min, aabb_max", "tv::array<T, 2>")
        code.arg("bound", "T") # 2 * log(alpha_eps / conic_opacity_vec[3])
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;

        bool valid = (std::get<1>(int_x0) >= aabb_min[1] && std::get<1>(int_x0) <= aabb_max[1]) || 
                     (std::get<2>(int_x0) >= aabb_min[1] && std::get<2>(int_x0) <= aabb_max[1]);
        valid |= (std::get<1>(int_x1) >= aabb_min[1] && std::get<1>(int_x1) <= aabb_max[1]) ||
                 (std::get<2>(int_x1) >= aabb_min[1] && std::get<2>(int_x1) <= aabb_max[1]);
        valid |= (std::get<1>(int_y0) >= aabb_min[0] && std::get<1>(int_y0) <= aabb_max[0]) ||
                    (std::get<2>(int_y0) >= aabb_min[0] && std::get<2>(int_y0) <= aabb_max[0]);
        valid |= (std::get<1>(int_y1) >= aabb_min[0] && std::get<1>(int_y1) <= aabb_max[0]) ||
                    (std::get<2>(int_y1) >= aabb_min[0] && std::get<2>(int_y1) <= aabb_max[0]);
        
        bool ellipse_cover_whole_aabb = std::get<1>(int_x0) <= aabb_min[1] && std::get<2>(int_x0) >= aabb_max[1];
        ellipse_cover_whole_aabb &= std::get<1>(int_x1) <= aabb_min[1] && std::get<2>(int_x1) >= aabb_max[1];
        ellipse_cover_whole_aabb &= std::get<1>(int_y0) <= aabb_min[0] && std::get<2>(int_y0) >= aabb_max[0];
        ellipse_cover_whole_aabb &= std::get<1>(int_y1) <= aabb_min[0] && std::get<2>(int_y1) >= aabb_max[0];
        
        bool aabb_cover_ellipse_center = aabb_min[0] <= center[0] && aabb_max[0] >= center[0] && aabb_min[1] <= center[1] && aabb_max[1] >= center[1];
        return valid || ellipse_cover_whole_aabb || aabb_cover_ellipse_center;
        """)
        return code.ret("bool")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def gs2d_ellipse_aabb_is_overlap_external_inter_v2(self):
        """ellipse: -0.5 * conic_opacity_vec[0] * (center[0] - x[0]) ^ 2 + 
                    -0.5 * conic_opacity_vec[2] * (center[1] - x[1]) ^ 2 -
                    conic_opacity_vec[1] * (center[0] - x[0]) * (center[1] - x[1])
                    = bound * 0.5
        bound: 2 * log(alpha_eps / conic_opacity_vec[3])

        calc ellipse is overlap with aabb

        VERY SLOW. consider `get_gaussian_2d_ellipse_bound_aabb` and `aabb_has_overlap_area`.
        """
        code = pccm.code()
        code.arg("int_x0, int_x1, int_y0, int_y1", "tv::array<int, 2>")
        code.arg("cur_tile_idx", "tv::array<int, 2>")

        code.raw(f"""

        bool valid = cur_tile_idx[1] >= int_x0[0] && cur_tile_idx[1] <= int_x0[1];
        valid |= cur_tile_idx[1] >= int_x1[0] && cur_tile_idx[1] <= int_x1[1];
        valid |= cur_tile_idx[0] >= int_y0[0] && cur_tile_idx[0] <= int_y0[1];
        valid |= cur_tile_idx[0] >= int_y1[0] && cur_tile_idx[0] <= int_y1[1];
        return valid;
        """)
        return code.ret("bool")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def solve_ellipse_y_constant(self):
        code = pccm.code()
        code.targ("T")
        code.arg("y", "T")
        code.arg("center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("bound", "T") # 2 * log(1 / (255 * conic_opacity_vec[3]))
        # 
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto cx = conic_opacity_vec[0];
        auto cy = conic_opacity_vec[1];
        auto cz = conic_opacity_vec[2];
        auto u = center[0];
        auto v = center[1];
        // from sympy, check __main in this file to get more info.
        // val_to_sqrt = -bound*cx - cx*cz*v**2 + 2*cx*cz*v*y - cx*cz*y**2 + cy**2*v**2 - 2*cy**2*v*y + cy**2*y**2
        auto val_to_sqrt =  -bound * cx - cx * cz * v * v + T(2) * cx * cz * v * y - cx * cz * y * y + cy * cy * v * v - T(2) * cy * cy * v * y + cy * cy * y * y;
        bool is_intersect = val_to_sqrt >= T(0);
        auto val_sqrted = math_op_t::fast_sqrt(math_op_t::max(T(0), val_to_sqrt));
        auto res_x = is_intersect ? (cx*u + cy*v - cy*y - val_sqrted) / cx : T(-1);
        auto res_y = is_intersect ? (cx*u + cy*v - cy*y + val_sqrted) / cx : T(-1);

        return std::make_tuple(is_intersect, res_x, res_y);
        """)
        return code.ret("std::tuple<bool, T, T>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def solve_ellipse_y_constant_v2(self):
        code = pccm.code()
        code.targ("T")
        code.arg("y", "T")
        code.arg("center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("bound", "T") # 2 * log(1 / (255 * conic_opacity_vec[3]))
        # 
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto cx = conic_opacity_vec[0];
        auto cy = conic_opacity_vec[1];
        auto cz = conic_opacity_vec[2];
        auto u = center[0];
        auto v = center[1];
        y -= v;
        // from sympy, check __main in this file to get more info.
        // val_to_sqrt = -bound*cx - cx*cz*y**2 + cy**2*y**2
        auto val_to_sqrt = -bound * cx - cx * cz * y * y + cy * cy * y * y;
        bool is_intersect = val_to_sqrt >= T(0);
        auto val_sqrted = math_op_t::fast_sqrt(math_op_t::max(T(0), val_to_sqrt));
        auto res_x = is_intersect ? u + (-cy * y - val_sqrted) / cx : T(-1);
        auto res_y = is_intersect ? u + (-cy * y + val_sqrted) / cx : T(-1);

        return std::make_tuple(is_intersect, res_x, res_y);
        """)
        return code.ret("std::tuple<bool, T, T>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def solve_ellipse_y_constant_tile(self):
        code = pccm.code()
        code.nontype_targ("TileDim", "int")
        code.targ("T")
        code.arg("y", "T")
        code.arg("center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("bound", "T") # 2 * log(1 / (255 * conic_opacity_vec[3]))
        # 
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto cx = conic_opacity_vec[0];
        auto cy = conic_opacity_vec[1];
        auto cz = conic_opacity_vec[2];
        auto u = center[0];
        auto v = center[1];
        y -= v;
        // from sympy, check __main in this file to get more info.
        // val_to_sqrt = -bound*cx - cx*cz*y**2 + cy**2*y**2
        auto val_to_sqrt = -bound * cx - cx * cz * y * y + cy * cy * y * y;
        bool is_intersect = val_to_sqrt >= T(0);
        auto val_sqrted = math_op_t::fast_sqrt(math_op_t::max(T(0), val_to_sqrt));
        auto res_x = is_intersect ? int(math_op_t::floor((u + (-cy * y - val_sqrted) / cx) / TileDim)) : int(-1);
        auto res_y = is_intersect ? int(math_op_t::floor((u + (-cy * y + val_sqrted) / cx) / TileDim)) : int(-1);

        return {{res_x, res_y}};
        """)
        return code.ret("tv::array<int, 2>")



    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def solve_ellipse_x_constant(self):
        code = pccm.code()
        code.targ("T")
        code.arg("x", "T")
        code.arg("center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("bound", "T") # 2 * log(1 / (255 * conic_opacity_vec[3]))
        # 
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto cx = conic_opacity_vec[0];
        auto cy = conic_opacity_vec[1];
        auto cz = conic_opacity_vec[2];
        auto u = center[0];
        auto v = center[1];
        // from sympy, check __main in this file to get more info.
        // val_to_sqrt = -bound*cz - cx*cz*u**2 + 2*cx*cz*u*x - cx*cz*x**2 + cy**2*u**2 - 2*cy**2*u*x + cy**2*x**2
        auto val_to_sqrt = -bound * cz - cx * cz * u * u + T(2) * cx * cz * u * x - cx * cz * x * x + cy * cy * u * u - T(2) * cy * cy * u * x + cy * cy * x * x;
        bool is_intersect = val_to_sqrt >= T(0);
        auto val_sqrted = math_op_t::fast_sqrt(math_op_t::max(T(0), val_to_sqrt));
        auto res_x = is_intersect ? (cy*u - cy*x + cz*v - val_sqrted) / cz : T(-1);
        auto res_y = is_intersect ? (cy*u - cy*x + cz*v + val_sqrted) / cz : T(-1);

        return std::make_tuple(is_intersect, res_x, res_y);
        """)
        return code.ret("std::tuple<bool, T, T>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def solve_ellipse_x_constant_v2(self):
        code = pccm.code()
        code.targ("T")
        code.arg("x", "T")
        code.arg("center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("bound", "T") # 2 * log(1 / (255 * conic_opacity_vec[3]))
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto cx = conic_opacity_vec[0];
        auto cy = conic_opacity_vec[1];
        auto cz = conic_opacity_vec[2];
        auto u = center[0];
        auto v = center[1];
        x -= u;
        // from sympy, check __main in this file to get more info.
        // val_to_sqrt = -bound*cz - cx*cz*x**2 + cy**2*x**2
        auto val_to_sqrt = -bound * cz - cx * cz * x * x + cy * cy * x * x;
        bool is_intersect = val_to_sqrt >= T(0);
        auto val_sqrted = math_op_t::fast_sqrt(math_op_t::max(T(0), val_to_sqrt));
        auto res_x = is_intersect ? v + (-cy * x - val_sqrted) / cz : T(-1);
        auto res_y = is_intersect ? v + (-cy * x + val_sqrted) / cz : T(-1);

        return std::make_tuple(is_intersect, res_x, res_y);
        """)
        return code.ret("std::tuple<bool, T, T>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def solve_ellipse_x_constant_tile(self):
        code = pccm.code()
        code.nontype_targ("TileDim", "int")
        code.targ("T")
        code.arg("x", "T")
        code.arg("center", "tv::array<T, 2>")
        code.arg("conic_opacity_vec", "tv::array<T, 4>")
        code.arg("bound", "T") # 2 * log(1 / (255 * conic_opacity_vec[3]))
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto cx = conic_opacity_vec[0];
        auto cy = conic_opacity_vec[1];
        auto cz = conic_opacity_vec[2];
        auto u = center[0];
        auto v = center[1];
        x -= u;
        // from sympy, check __main in this file to get more info.
        // val_to_sqrt = -bound*cz - cx*cz*x**2 + cy**2*x**2
        auto val_to_sqrt = -bound * cz - cx * cz * x * x + cy * cy * x * x;
        bool is_intersect = val_to_sqrt >= T(0);
        auto val_sqrted = math_op_t::fast_sqrt(math_op_t::max(T(0), val_to_sqrt));
        auto res_x = is_intersect ? int(math_op_t::floor((v + (-cy * x - val_sqrted) / cz) / TileDim)) : int(-1);
        auto res_y = is_intersect ? int(math_op_t::floor((v + (-cy * x + val_sqrted) / cz) / TileDim)) : int(-1);

        return {{res_x, res_y}};
        """)
        return code.ret("tv::array<int, 2>")



    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def aabb_has_overlap_area(self):
        code = pccm.code()
        code.targ("T")
        code.arg("aabb_min1, aabb_max1, aabb_min2, aabb_max2", "tv::array<T, 2>")
        # 
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        bool x_overlap = aabb_min1[0] <= aabb_max2[0] && aabb_max1[0] >= aabb_min2[0];
        bool y_overlap = aabb_min1[1] <= aabb_max2[1] && aabb_max1[1] >= aabb_min2[1];
        return x_overlap && y_overlap;
        """)
        return code.ret("bool")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def aabb_overlap_obb(self):
        code = pccm.code()
        code.targ("T")
        code.arg("aabb_min, aabb_max", "tv::array<T, 2>")
        code.arg("obb_whsc", "tv::array<T, 4>")
        code.arg("obb_center", "tv::array<T, 2>")
        """

        cocos2d::Vec2 a1( cos(o1.rotation), sin(o1.rotation));
        cocos2d::Vec2 a2(-sin(o1.rotation), cos(o1.rotation));
        cocos2d::Vec2 a3( cos(o2.rotation), sin(o2.rotation));
        cocos2d::Vec2 a4(-sin(o2.rotation), cos(o2.rotation));

        // edge length
        cocos2d::Size l1 = o1.size * 0.5f;
        cocos2d::Size l2 = o2.size * 0.5f;

        // vector between pivots
        cocos2d::Vec2 l = o1.pivot - o2.pivot;

        float r1, r2, r3, r4;

        // project to a1
        r1 = l1.width  * fabs(a1.dot(a1));
        r2 = l1.height * fabs(a2.dot(a1));
        r3 = l2.width  * fabs(a3.dot(a1));
        r4 = l2.height * fabs(a4.dot(a1));
        if (r1 + r2 + r3 + r4 <= fabs(l.dot(a1)))
        {
            return false;
        }

        // project to a2
        r1 = l1.width  * fabs(a1.dot(a2));
        r2 = l1.height * fabs(a2.dot(a2));
        r3 = l2.width  * fabs(a3.dot(a2));
        r4 = l2.height * fabs(a4.dot(a2));
        if (r1 + r2 + r3 + r4 <= fabs(l.dot(a2)))
        {
            return false;
        }

        // project to a3
        r1 = l1.width  * fabs(a1.dot(a3));
        r2 = l1.height * fabs(a2.dot(a3));
        r3 = l2.width  * fabs(a3.dot(a3));
        r4 = l2.height * fabs(a4.dot(a3));
        if (r1 + r2 + r3 + r4 <= fabs(l.dot(a3)))
        {
            return false;
        }

        // project to a4
        r1 = l1.width  * fabs(a1.dot(a4));
        r2 = l1.height * fabs(a2.dot(a4));
        r3 = l2.width  * fabs(a3.dot(a4));
        r4 = l2.height * fabs(a4.dot(a4));
        if (r1 + r2 + r3 + r4 <= fabs(l.dot(a4)))
        {
            return false;
        }

        return true;

        """
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto sin_theta = obb_whsc[2];
        auto cos_theta = obb_whsc[3];
        auto w = obb_whsc[0];
        auto h = obb_whsc[1];
        auto center = obb_center;
        // o1: aabb, o2: obb
        tv::array<T, 2> a1{{1, 0}};
        tv::array<T, 2> a2{{0, 1}};
        tv::array<T, 2> a3{{cos_theta, sin_theta}};
        tv::array<T, 2> a4{{-sin_theta, cos_theta}};

        tv::array<T, 2> l1 = (aabb_max - aabb_min) / T(2);
        tv::array<T, 2> l2{{w / T(2), h / T(2)}};

        tv::array<T, 2> l = (aabb_max + aabb_min) / T(2) - center;
        float r1, r2, r3, r4;
        r1 = l1[0]  * math_op_t::abs(a1.template op<op::dot>(a1));
        r2 = l1[1] * math_op_t::abs(a2.template op<op::dot>(a1));
        r3 = l2[0]  * math_op_t::abs(a3.template op<op::dot>(a1));
        r4 = l2[1] * math_op_t::abs(a4.template op<op::dot>(a1));
        if (r1 + r2 + r3 + r4 <= math_op_t::abs(l.template op<op::dot>(a1)))
        {{
            return false;
        }}

        r1 = l1[0]  * math_op_t::abs(a1.template op<op::dot>(a2));
        r2 = l1[1] * math_op_t::abs(a2.template op<op::dot>(a2));
        r3 = l2[0]  * math_op_t::abs(a3.template op<op::dot>(a2));
        r4 = l2[1] * math_op_t::abs(a4.template op<op::dot>(a2));
        if (r1 + r2 + r3 + r4 <= math_op_t::abs(l.template op<op::dot>(a2)))
        {{
            return false;
        }}

        r1 = l1[0]  * math_op_t::abs(a1.template op<op::dot>(a3));
        r2 = l1[1] * math_op_t::abs(a2.template op<op::dot>(a3));
        r3 = l2[0]  * math_op_t::abs(a3.template op<op::dot>(a3));
        r4 = l2[1] * math_op_t::abs(a4.template op<op::dot>(a3));
        if (r1 + r2 + r3 + r4 <= math_op_t::abs(l.template op<op::dot>(a3)))
        {{
            return false;
        }}

        r1 = l1[0]  * math_op_t::abs(a1.template op<op::dot>(a4));
        r2 = l1[1] * math_op_t::abs(a2.template op<op::dot>(a4));
        r3 = l2[0]  * math_op_t::abs(a3.template op<op::dot>(a4));
        r4 = l2[1] * math_op_t::abs(a4.template op<op::dot>(a4));
        if (r1 + r2 + r3 + r4 <= math_op_t::abs(l.template op<op::dot>(a4)))
        {{
            return false;
        }}

        return true;
        
        """)
        return code.ret("bool")


def __main():

    op = GaussianEllipseOp().build()
    op_centered = GaussianEllipseCenterOp().build()

    res = op.name_to_res_sym["res"]
    print(res.sym)
    print(sympy.simplify(sympy.solveset(sympy.Eq(res.sym, 0), op.name_to_sym["y"].sym)))
    print(sympy.simplify(sympy.solveset(sympy.Eq(res.sym, 0), op.name_to_sym["x"].sym)))

    res = op_centered.name_to_res_sym["res"]
    print("-------")
    print(res.sym)
    print(sympy.simplify(sympy.solveset(sympy.Eq(res.sym, 0), op_centered.name_to_sym["y"].sym)))
    print(sympy.simplify(sympy.solveset(sympy.Eq(res.sym, 0), op_centered.name_to_sym["x"].sym)))
    print("-------")
    op_bound = EllipseBoundOp().build()
    res_x = op_bound.name_to_res_sym["res_x"]
    res_y = op_bound.name_to_res_sym["res_y"]

    print(sympy.solveset(sympy.Eq(res_x.sym, 0), op_bound.name_to_sym["x"].sym))
    print(sympy.solveset(sympy.Eq(res_y.sym, 0), op_bound.name_to_sym["y"].sym))

    pass 

if __name__ == "__main__":
    __main()