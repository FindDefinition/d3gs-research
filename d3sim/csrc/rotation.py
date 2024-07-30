import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC

class RotationMath(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC)

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
        return {seq[0]}.template op<op::mm_tt>({seq[1]}.template op<op::mm_tt>({seq[2]}));
        """)
        return code.ret("tv::array_nd<T, 3, 3>")

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def euler_to_rotmat_zyx(self):
        return self.euler_to_rotmat_template(["z", "y", "x"])

    @pccm.cuda.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def euler_to_rotmat_xyz(self):
        return self.euler_to_rotmat_template(["x", "y", "z"])