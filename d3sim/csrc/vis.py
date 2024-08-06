    
import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC


class VisUtils(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC)

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def value_to_jet_rgb(self):
        code = pccm.code()
        code.targ("T")
        code.arg("val, lower, upper", "T")
        code.raw(f"""
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        T t = (val - lower) / (upper - lower);
        t = math_op_t::clamp(t, 0, 1);
        float r = 0;
        float g = 0;
        float b = 0;
        if (t <= 0.125f) {{
            r = 0;
            g = 0;
            b = 0.5f + 4 * t;
        }} else if (t <= 0.375f) {{
            r = 0;
            g = 4 * t - 0.5f;
            b = 1;
        }} else if (t <= 0.625f) {{
            r = 4 * t - 1.5f;
            g = 1;
            b = 1.5f - 4 * t;
        }} else if (t <= 0.875f) {{
            r = 1;
            g = 1.5f - 4 * t;
            b = 0;
        }} else {{
            r = 1.5f - 4 * t;
            g = 0;
            b = 0;
        }}
        // convert to 0-255
        uint8_t ur = std::max(0, std::min(255, int(r * 255)));
        uint8_t ug = std::max(0, std::min(255, int(g * 255)));
        uint8_t ub = std::max(0, std::min(255, int(b * 255)));
        return tv::array<uint8_t, 3>{{ur, ug, ub}};
        """)
        return code.ret("tv::array<uint8_t, 3>")
