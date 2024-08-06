    
import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC

class CameraDefs(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_enum("DistortType", [("kOpencvPinhole", 0)])

    @pccm.pybind.mark
    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def hello(self):
        return pccm.code()

class CameraOps(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC, CameraDefs)

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def camera_distortion_pixel(self):
        code = pccm.code()
        code.targ("T")
        code.arg("k1k2p1p2", "TV_METAL_THREAD const T*")
        code.arg("u,v", "T")
        code.arg("du,dv", "TV_METAL_THREAD T*")
        code.raw(f"""
        const T k1 = k1k2p1p2[0];
        const T k2 = k1k2p1p2[1];
        const T p1 = k1k2p1p2[2];
        const T p2 = k1k2p1p2[3];
        const T u2 = u * u;
        const T uv = u * v;
        const T v2 = v * v;
        const T r2 = u2 + v2;
        const T radial = k1 * r2 + k2 * r2 * r2;
        *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
        *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
        return true;
        """)
        return code.ret("bool")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_to_uv(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")

        code.arg("pos", "tv::array<float, 3>")
        code.arg("resolution", "tv::array<float, 2>")
        code.arg("principal_point", "tv::array<float, 2>")
        code.arg("focal_length", "tv::array<float, 2>")
        code.arg("distort", "const TV_METAL_THREAD float*")
        code.arg("distort_type", "CameraDefs::DistortType")
        code.arg("cam2world_T", "const TV_METAL_THREAD tv::array<tv::array<float, 3>, 4>&")
        code.arg("parallax_shift", "tv::array<float, 3>")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = op::MathScalarOp<float>;
        using math_op_int_t = op::MathScalarOp<int>;

        float AxisFrontSign = math_op_t::copysign(1.0f, AxisFront);
        float AxisUSign = math_op_t::copysign(1.0f, AxisU);
        float AxisVSign = math_op_t::copysign(1.0f, AxisV);
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        using vec3 = tv::array<float, 3>;
        using vec2 = tv::array<float, 2>;
                 
        vec3 head_pos = {{parallax_shift[0], parallax_shift[1], 0.f}};
        vec3 origin =  op::slice<0, 3>(cam2world_T).op<op::mv_colmajor>(head_pos) + cam2world_T[3];
        vec3 dir = pos - origin;
        // dir = op::slice<0, 3>(cam2world_T).op<op::inverse>().op<op::mv_colmajor>(dir);
        dir = op::slice<0, 3>(cam2world_T).op<op::mv_rowmajor>(dir);
        float z_in_cam = AxisFrontSign * dir[AxisFrontAbs];
        dir[AxisUAbs] *= AxisUSign;
        dir[AxisVAbs] *= AxisVSign;

        dir /= float(dir[AxisFrontAbs]);
        dir += head_pos * parallax_shift[AxisFront];
        float du = 0.0f, dv = 0.0f;
        bool valid = true;
        if (distort_type == CameraDefs::DistortType::kOpencvPinhole){{
            valid = camera_distortion_pixel(distort, dir[AxisUAbs], dir[AxisVAbs], &du, &dv);
        }}
        dir[AxisUAbs] += du;
        dir[AxisVAbs] += dv;
        vec2 uv{{dir[AxisUAbs], dir[AxisVAbs]}};
        uv = uv * focal_length / resolution + principal_point;
        
        return std::make_tuple(uv, z_in_cam, valid);
        """)
        return code.ret("std::tuple<tv::array<float, 2>, float, bool>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_to_uv_no_distort(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")

        code.arg("pos", "tv::array<float, 3>")
        code.arg("principal_point", "tv::array<float, 2>")
        code.arg("focal_length_unified", "tv::array<float, 2>")
        code.arg("cam2world_T", "const TV_METAL_THREAD tv::array<tv::array<float, 3>, 4>&")
        code.raw(f"""
        namespace op = tv::arrayops;

        constexpr int AxisFrontSign = AxisFront > 0 ? 1 : -1;
        constexpr int AxisUSign = AxisU > 0 ? 1 : -1;
        constexpr int AxisVSign = AxisV > 0 ? 1 : -1;
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        using vec3 = tv::array<float, 3>;
        using vec2 = tv::array<float, 2>;
        vec3 dir = pos - cam2world_T[3];
        dir = op::slice<0, 3>(cam2world_T).op<op::mv_rowmajor>(dir);
        vec2 uv{{AxisUSign * dir[AxisUAbs] / dir[AxisFrontAbs], AxisVSign * dir[AxisVAbs] / dir[AxisFrontAbs]}};
        uv = uv * focal_length_unified + principal_point;
        return std::make_tuple(uv, AxisFrontSign * dir[AxisFrontAbs]);
        """)
        return code.ret("std::tuple<tv::array<float, 2>, float>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_cam_to_uv_no_distort(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")

        code.arg("pos_cam", "tv::array<float, 3>")
        code.arg("principal_point", "tv::array<float, 2>")
        code.arg("focal_length_unified", "tv::array<float, 2>")
        code.raw(f"""
        namespace op = tv::arrayops;
        constexpr int AxisFrontSign = AxisFront > 0 ? 1 : -1;
        constexpr int AxisUSign = AxisU > 0 ? 1 : -1;
        constexpr int AxisVSign = AxisV > 0 ? 1 : -1;
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        using vec3 = tv::array<float, 3>;
        using vec2 = tv::array<float, 2>;
        vec2 uv{{AxisUSign * pos_cam[AxisUAbs] / pos_cam[AxisFrontAbs], AxisVSign * pos_cam[AxisVAbs] / pos_cam[AxisFrontAbs]}};
        uv = uv * focal_length_unified + principal_point;
        return std::make_tuple(uv, AxisFrontSign * pos_cam[AxisFrontAbs]);
        """)
        return code.ret("std::tuple<tv::array<float, 2>, float>")


    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_cam_to_ndc_uv_no_distort(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")
        code.nontype_targ("AxisFrontSign", "int", "1")
        code.nontype_targ("AxisUSign", "int", "1")
        code.nontype_targ("AxisVSign", "int", "1")

        code.arg("pos_cam", "tv::array<float, 3>")
        code.arg("principal_point", "tv::array<float, 2>")
        code.arg("focal_length_unified", "tv::array<float, 2>")
        code.raw(f"""
        namespace op = tv::arrayops;
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        using vec3 = tv::array<float, 3>;
        using vec2 = tv::array<float, 2>;
        vec2 uv{{AxisUSign * pos_cam[AxisUAbs] / pos_cam[AxisFrontAbs], AxisVSign * pos_cam[AxisVAbs] / pos_cam[AxisFrontAbs]}};
        uv = uv * 2 * focal_length_unified;
        return std::make_tuple(uv, AxisFrontSign * pos_cam[AxisFrontAbs]);
        """)
        return code.ret("std::tuple<tv::array<float, 2>, float>")
