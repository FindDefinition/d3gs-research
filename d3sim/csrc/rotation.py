import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC
from d3sim.core.geodef import EulerIntrinsicOrder


class RotationMath(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC)
        self.add_enum("EulerIntrinsicOrder", [
            ("kXYZ", EulerIntrinsicOrder.XYZ.value),
            ("kZYX", EulerIntrinsicOrder.ZYX.value),
            ("kZXY", EulerIntrinsicOrder.ZXY.value),
            ("kXZY", EulerIntrinsicOrder.XZY.value),
            ("kYXZ", EulerIntrinsicOrder.YXZ.value),
            ("kYZX", EulerIntrinsicOrder.YZX.value),

        ])

    def euler_to_rotmat_template(self, seq: list[str]):
        code = pccm.code()
        code.targ("T")
        code.arg("roll, pitch, yaw", "T")
        code.raw("""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        T cos_roll = math_op_t::cos(roll), cos_pitch = math_op_t::cos(pitch), cos_yaw = math_op_t::cos(yaw),
            sin_roll = math_op_t::sin(roll), sin_pitch = math_op_t::sin(pitch), sin_yaw = math_op_t::sin(yaw);

        tv::array_nd<T, 3, 3> x{
            tv::array<T, 3>{1, 0, 0},
            tv::array<T, 3>{0, cos_roll, -sin_roll},
            tv::array<T, 3>{0, sin_roll, cos_roll}
        };
        tv::array_nd<T, 3, 3> y{
            tv::array<T, 3>{cos_pitch, 0, sin_pitch},
            tv::array<T, 3>{0, 1, 0},
            tv::array<T, 3>{-sin_pitch, 0, cos_pitch}
        };
        tv::array_nd<T, 3, 3> z{
            tv::array<T, 3>{cos_yaw, -sin_yaw, 0},
            tv::array<T, 3>{sin_yaw, cos_yaw, 0},
            tv::array<T, 3>{0, 0, 1}
        };

        """)
        code.raw(f"""
        return {seq[0]}.template op<op::mm_ttt>({seq[1]}.template op<op::mm_ttt>({seq[2]}));
        """)
        return code.ret("tv::array_nd<T, 3, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def euler_to_rotmat_zyx(self):
        return self.euler_to_rotmat_template(["z", "y", "x"])

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def euler_to_rotmat_xyz(self):
        return self.euler_to_rotmat_template(["x", "y", "z"])
    
    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def clamp(self):
        code = pccm.code()
        code.targ("T")
        code.arg("val", "T")
        code.arg("min_val", "T")
        code.arg("max_val", "T")
        code.raw(f"""
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        return math_op_t::max(min_val, math_op_t::min(max_val, val));
        """)
        return code.ret("T")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def rotmat_to_euler(self):
        code = pccm.code()
        code.targ("T")
        code.arg("rot", "tv::array_nd<T, 3, 3>")
        code.arg("order", "EulerIntrinsicOrder")
        code.raw("""
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        auto m11 = rot[0][0], m12 = rot[0][1], m13 = rot[0][2];
        auto m21 = rot[1][0], m22 = rot[1][1], m23 = rot[1][2];
        auto m31 = rot[2][0], m32 = rot[2][1], m33 = rot[2][2];
        T roll = 0;
        T pitch = 0;
        T yaw = 0;
        switch (order) {
            case EulerIntrinsicOrder::kXYZ: {
                pitch = math_op_t::asin(clamp(m13, -1, 1));
                if (math_op_t::abs(m13) < 0.9999999) {
                    roll = math_op_t::atan2(-m23, m33);
                    yaw = math_op_t::atan2(-m12, m11);
                } else {
                    roll = math_op_t::atan2(m32, m22);
                    yaw = 0;
                }
                break;    
            }
            case EulerIntrinsicOrder::kZYX: {
                pitch = math_op_t::asin(-clamp(m31, -1, 1));
                if (math_op_t::abs(m31) < 0.9999999) {
                    roll = math_op_t::atan2(m32, m33);
                    yaw = math_op_t::atan2(m21, m11);
                } else {
                    roll = 0;
                    yaw = math_op_t::atan2(-m12, m22);
                }
                break;
            }
            case EulerIntrinsicOrder::kZXY: {
                pitch = math_op_t::asin(clamp(m32, -1, 1));
                if (math_op_t::abs(m32) < 0.9999999) {
                    roll = math_op_t::atan2(-m31, m33);
                    yaw = math_op_t::atan2(-m12, m22);
                } else {
                    roll = 0;
                    yaw = math_op_t::atan2(m21, m11);
                }
                break;
            }
            case EulerIntrinsicOrder::kXZY: {
                pitch = math_op_t::asin(-clamp(m12, -1, 1));
                if (math_op_t::abs(m12) < 0.9999999) {
                    roll = math_op_t::atan2(m32, m22);
                    yaw = math_op_t::atan2(m13, m11);
                } else {
                    roll = math_op_t::atan2(-m23, m33);
                    yaw = 0;
                }
                break;
            }
            case EulerIntrinsicOrder::kYXZ: {
                pitch = math_op_t::asin(-clamp(m23, -1, 1));
                if (math_op_t::abs(m23) < 0.9999999) {
                    roll = math_op_t::atan2(m13, m33);
                    yaw = math_op_t::atan2(m21, m22);
                } else {
                    roll = math_op_t::atan2(-m31, m11);
                    yaw = 0;
                }
                break;
            }
            case EulerIntrinsicOrder::kYZX: {
                yaw = math_op_t::asin(clamp(m21, -1, 1));
                if (math_op_t::abs(m21) < 0.9999999) {
                    roll = math_op_t::atan2(-m23, m22);
                    pitch = math_op_t::atan2(-m31, m11);
                } else {
                    roll = 0;
                    pitch = math_op_t::atan2(m13, m33);
                }
                break;
            }
            default:;
        }
        return tv::array<T, 3>{roll, pitch, yaw};
        """)
        return code.ret("tv::array<T, 3>")

