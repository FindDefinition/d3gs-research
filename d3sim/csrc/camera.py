    
import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC

class CameraDefs(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_enum("DistortType", [
            ("kNone", 0),
            ("kOpencvPinhole", 1),
            ("kOpencvPinholeWaymo", 2),
        ])

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

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def camera_distortion_pixel_waymo(self):
        code = pccm.code()
        code.targ("T")
        code.arg("k1k2p1p2", "TV_METAL_THREAD const T*")
        code.arg("u,v", "T")
        code.arg("du,dv", "TV_METAL_THREAD T*")

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = op::MathScalarOp<T>;
        const T k1 = k1k2p1p2[0];
        const T k2 = k1k2p1p2[1];
        const T p1 = k1k2p1p2[2];
        const T p2 = k1k2p1p2[3];
        const T u2 = u * u;
        const T uv = u * v;
        const T v2 = v * v;
        const T r2 = u2 + v2;
        const T radial = k1 * r2 + k2 * r2 * r2;
        bool valid = true;
        if (radial < -0.2f || radial > 0.2f) {{
            valid = false;
        }}
        *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
        *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2) + T(2) * p2 * uv;
        return valid;
        """)
        return code.ret("bool")

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def inverse_2x2(self):
        code = pccm.code()
        code.targ("T")
        code.arg("mat", "TV_METAL_THREAD const T*")
        code.arg("out_mat", "TV_METAL_THREAD T*")
        code.raw(f"""
        T det = mat[0] * mat[3] - mat[1] * mat[2] + 1e-10;
        out_mat[0] = mat[3] / det;
        out_mat[1] = -mat[1] / det;
        out_mat[2] = -mat[2] / det;
        out_mat[3] = mat[0] / det;
        """)
        return code 

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def iterative_camera_undistortion_pixel_template(self):
        code = pccm.code()
        code.targ("T")
        code.targ("F")
        code.arg("k1k2p1p2", "const TV_METAL_THREAD T*")
        code.arg("u,v", "TV_METAL_THREAD T*")
        code.arg("distortion_fun", "F")

        code.raw(f"""
        // from instant-ngp.
        constexpr uint32_t kNumIterations = 100;
        constexpr T kMaxStepNorm = 1e-10f;
        constexpr T kRelStepSize = 1e-6f;
        T x[2], x0[2], dx[2], dx_0b[2], dx_0f[2], dx_1b[2], dx_1f[2], J[4], J_inv[4];
        x[0] = *u;
        x[1] = *v;
        x0[0] = *u;
        x0[1] = *v;

        for (uint32_t i = 0; i < kNumIterations; ++i) {{
            const T step0 = std::max(std::numeric_limits<T>::epsilon(), std::abs(kRelStepSize * x[0]));
            const T step1 = std::max(std::numeric_limits<T>::epsilon(), std::abs(kRelStepSize * x[1]));
            distortion_fun(k1k2p1p2, x[0], x[1], &dx[0], &dx[1]);
            distortion_fun(k1k2p1p2, x[0] - step0, x[1], &dx_0b[0], &dx_0b[1]);
            distortion_fun(k1k2p1p2, x[0] + step0, x[1], &dx_0f[0], &dx_0f[1]);
            distortion_fun(k1k2p1p2, x[0], x[1] - step1, &dx_1b[0], &dx_1b[1]);
            distortion_fun(k1k2p1p2, x[0], x[1] + step1, &dx_1f[0], &dx_1f[1]);
            J[0] = 1 + (dx_0f[0] - dx_0b[0]) / (2 * step0);
            J[1] = (dx_1f[0] - dx_1b[0]) / (2 * step1);
            J[2] = (dx_0f[1] - dx_0b[1]) / (2 * step0);
            J[3] = 1 + (dx_1f[1] - dx_1b[1]) / (2 * step1);
            inverse_2x2(&J[0], &J_inv[0]);
            T x_dx_x0[2] = {{x[0] + dx[0] - x0[0], x[1] + dx[1] - x0[1]}};
            T step_x[2] = {{J_inv[0] * x_dx_x0[0] + J_inv[1] * x_dx_x0[1], 
                J_inv[2] * x_dx_x0[0] + J_inv[3] * x_dx_x0[1]}};
            x[0] -= step_x[0];
            x[1] -= step_x[1];
            if (step_x[0] * step_x[0] + step_x[1] * step_x[1] < kMaxStepNorm) {{
                break;
            }}
        }}
        *u = x[0];
        *v = x[1];
        return true;
        """)
        return code.ret("bool")
    
    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def iterative_camera_undistortion_pixel_waymo(self):
        code = pccm.code()
        code.targ("T")
        code.arg("k1k2p1p2", "TV_METAL_THREAD const T*")
        code.arg("u_ptr, v_ptr", "TV_METAL_THREAD T*")
        code.arg("focal_length", "tv::array<T, 2>")

        code.raw("""
        namespace op = tv::arrayops;
        const T eps = T(1e-12) / (focal_length.template op<op::length2>());
        const T k1 = k1k2p1p2[0];
        const T k2 = k1k2p1p2[1];
        const T p1 = k1k2p1p2[2];
        const T p2 = k1k2p1p2[3];

        constexpr int kNumIterations = 20;
        auto u = *u_ptr;
        auto v = *v_ptr;
        auto u_init = *u_ptr;
        auto v_init = *v_ptr;

        for (int i = 0; i < kNumIterations; ++i) {
            const T u2 = u * u;
            const T uv = u * v;
            const T v2 = v * v;
            const T r2 = u2 + v2;
            const T radial = T(1) + k1 * r2 + k2 * r2 * r2;
            const T u_prev = u;
            const T v_prev = v;

            const T u_tangential = T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
            const T v_tangential = T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
            u = (u_init - u_tangential) / radial;
            v = (v_init - v_tangential) / radial;

            const T du = u - u_prev;
            const T dv = v - v_prev;
            // Early exit.
            if (du * du + dv * dv < eps) {
                break;
            }
        }
        *u_ptr = u;
        *v_ptr = v;
        return true;
        """)
        return code.ret("bool")

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def iterative_camera_undistortion_pixel(self):
        code = pccm.code()
        code.targ("T")
        code.arg("k1k2p1p2", "TV_METAL_THREAD const T*")
        code.arg("u,v", "TV_METAL_THREAD T*")
        code.raw(f"""
        return iterative_camera_undistortion_pixel_template(k1k2p1p2, u, v, camera_distortion_pixel<T>);
        """)
        return code.ret("bool")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def uv_to_dir(self):
        code = pccm.code()
        code.targ("T")
        code.arg("uv_pixel", "tv::array<T, 2>")
        code.arg("principal_point", "tv::array<T, 2>")
        code.arg("focal_length", "tv::array<T, 2>")
        code.arg("distort", "const TV_METAL_THREAD T*")
        code.arg("distort_type", "CameraDefs::DistortType")
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = op::MathScalarOp<T>;
        uv_pixel = (uv_pixel - principal_point) / focal_length;
        if (distort_type == CameraDefs::DistortType::kOpencvPinholeWaymo){{
            iterative_camera_undistortion_pixel_waymo(distort, &uv_pixel[0], &uv_pixel[1], focal_length);
        }} else if (distort_type == CameraDefs::DistortType::kOpencvPinhole){{
            iterative_camera_undistortion_pixel(distort, &uv_pixel[0], &uv_pixel[1]);
        }}
        return {{uv_pixel[0], uv_pixel[1], T(1)}};
        """)
        return code.ret("tv::array<T, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_to_uv(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")
        code.nontype_targ("AxisFrontSign", "int", "1")
        code.nontype_targ("AxisUSign", "int", "1")
        code.nontype_targ("AxisVSign", "int", "1")

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

        using vec3 = tv::array<float, 3>;
        using vec2 = tv::array<float, 2>;
                 
        vec3 head_pos = {{parallax_shift[0], parallax_shift[1], 0.f}};
        vec3 origin =  op::slice<0, 3>(cam2world_T).op<op::mv_colmajor>(head_pos) + cam2world_T[3];
        vec3 dir = pos - origin;
        // dir = op::slice<0, 3>(cam2world_T).op<op::inverse>().op<op::mv_colmajor>(dir);
        dir = op::slice<0, 3>(cam2world_T).op<op::mv_rowmajor>(dir);
        float z_in_cam = AxisFrontSign * dir[AxisFront];
        dir[AxisU] *= AxisUSign;
        dir[AxisV] *= AxisVSign;

        dir /= float(dir[AxisFront]);
        dir += head_pos * parallax_shift[AxisFront];
        float du = 0.0f, dv = 0.0f;
        bool valid = true;
        if (distort_type == CameraDefs::DistortType::kOpencvPinhole){{
            valid = camera_distortion_pixel(distort, dir[AxisU], dir[AxisV], &du, &dv);
        }} else if (distort_type == CameraDefs::DistortType::kOpencvPinholeWaymo){{
            valid = camera_distortion_pixel_waymo(distort, dir[AxisU], dir[AxisV], &du, &dv);
        }}
        dir[AxisU] += du;
        dir[AxisV] += dv;
        vec2 uv{{dir[AxisU], dir[AxisV]}};
        uv = uv * focal_length / resolution + principal_point;
        
        return std::make_tuple(uv, z_in_cam, valid);
        """)
        return code.ret("std::tuple<tv::array<float, 2>, float, bool>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_to_uv_no_distort(self):
        code = pccm.code()
        code.targ("T")
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")
        code.nontype_targ("AxisFrontSign", "int", "1")
        code.nontype_targ("AxisUSign", "int", "1")
        code.nontype_targ("AxisVSign", "int", "1")

        code.arg("pos", "tv::array<T, 3>")
        code.arg("principal_point", "tv::array<T, 2>")
        code.arg("focal_length_unified", "tv::array<T, 2>")
        code.arg("cam2world_T", "const TV_METAL_THREAD tv::array<tv::array<T, 3>, 4>&")
        code.raw(f"""
        namespace op = tv::arrayops;
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        using vec3 = tv::array<T, 3>;
        using vec2 = tv::array<T, 2>;
        vec3 dir = pos - cam2world_T[3];
        dir = op::slice<0, 3>(cam2world_T).template op<op::mv_rowmajor>(dir);
        vec2 uv{{AxisUSign * dir[AxisUAbs] / dir[AxisFrontAbs], AxisVSign * dir[AxisVAbs] / dir[AxisFrontAbs]}};
        uv = uv * focal_length_unified + principal_point;
        return std::make_tuple(uv, AxisFrontSign * dir[AxisFrontAbs]);
        """)
        return code.ret("std::tuple<tv::array<T, 2>, T>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_cam_to_uv_no_distort(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")
        code.nontype_targ("AxisFrontSign", "int", "1")
        code.nontype_targ("AxisUSign", "int", "1")
        code.nontype_targ("AxisVSign", "int", "1")
        code.targ("T")

        code.arg("pos_cam", "tv::array<T, 3>")
        code.arg("principal_point", "tv::array<T, 2>")
        code.arg("focal_length_unified", "tv::array<T, 2>")
        code.raw(f"""
        namespace op = tv::arrayops;
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        using vec3 = tv::array<T, 3>;
        using vec2 = tv::array<T, 2>;
        vec2 uv{{AxisUSign * pos_cam[AxisUAbs] / pos_cam[AxisFrontAbs], AxisVSign * pos_cam[AxisVAbs] / pos_cam[AxisFrontAbs]}};
        uv = uv * focal_length_unified + principal_point;
        return std::make_tuple(uv, AxisFrontSign * pos_cam[AxisFrontAbs]);
        """)
        return code.ret("std::tuple<tv::array<T, 2>, T>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_cam_to_uv_no_distort_grad(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")
        code.nontype_targ("AxisFrontSign", "int", "1")
        code.nontype_targ("AxisUSign", "int", "1")
        code.nontype_targ("AxisVSign", "int", "1")
        code.targ("T")
        code.arg("duv", "tv::array<T, 2>")
        code.arg("dz", "T")

        code.arg("pos_cam", "tv::array<T, 3>")
        code.arg("principal_point", "tv::array<T, 2>")
        code.arg("focal_length", "tv::array<T, 2>")
        code.raw(f"""
        namespace op = tv::arrayops;
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        duv = duv * focal_length;
        auto origin_z_inv = 1.0f / (pos_cam[AxisFrontAbs]);

        auto origin_z_inv2 = 1.0f / (pos_cam[AxisFrontAbs] * pos_cam[AxisFrontAbs]);

        tv::array<T, 3> res;
        res[AxisUAbs] = AxisUSign * duv[0] * origin_z_inv;
        res[AxisVAbs] = AxisVSign * duv[1] * origin_z_inv;
        res[AxisFrontAbs] = (AxisFrontSign * dz + 
            -AxisUSign * duv[0] * pos_cam[AxisUAbs] * origin_z_inv2 +
            -AxisVSign * duv[1] * pos_cam[AxisVAbs] * origin_z_inv2);
        
        return res;
        """)
        return code.ret("tv::array<T, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def pos_cam_to_ndc_uv_no_distort(self):
        code = pccm.code()
        code.nontype_targ("AxisFront", "int", "2")
        code.nontype_targ("AxisU", "int", "0")
        code.nontype_targ("AxisV", "int", "1")
        code.nontype_targ("AxisFrontSign", "int", "1")
        code.nontype_targ("AxisUSign", "int", "1")
        code.nontype_targ("AxisVSign", "int", "1")
        code.targ("T")

        code.arg("pos_cam", "tv::array<T, 3>")
        code.arg("principal_point", "tv::array<T, 2>")
        code.arg("focal_length_unified", "tv::array<T, 2>")
        code.raw(f"""
        namespace op = tv::arrayops;
        constexpr int AxisFrontAbs = AxisFront > 0 ? AxisFront : -AxisFront;
        constexpr int AxisUAbs = AxisU > 0 ? AxisU : -AxisU;
        constexpr int AxisVAbs = AxisV > 0 ? AxisV : -AxisV;

        using vec3 = tv::array<T, 3>;
        using vec2 = tv::array<T, 2>;
        vec2 uv{{AxisUSign * pos_cam[AxisUAbs] / pos_cam[AxisFrontAbs], AxisVSign * pos_cam[AxisVAbs] / pos_cam[AxisFrontAbs]}};
        uv = uv * 2 * focal_length_unified;
        return std::make_tuple(uv, AxisFrontSign * pos_cam[AxisFrontAbs]);
        """)
        return code.ret("std::tuple<tv::array<T, 2>, T>")
