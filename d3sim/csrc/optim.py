    
import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC

class OptimOps(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC)

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def is_aligned_ptr(self):
        code = pccm.code()
        code.nontype_targ("N", "size_t")
        code.targ("T")
        code.arg("ptr", "TV_METAL_DEVICE T*")
        code.raw(f"""
        return ((uint64_t)ptr) % (N * sizeof(T)) == 0;
        """)
        return code.ret("bool")


    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def adam(self):
        code = pccm.code()
        code.targ("T")
        code.nontype_targ("IsAdamW", "bool", "false")
        # input only
        code.arg("grad", "T")
        # input and output
        code.arg("param, exp_avg, exp_avg_sq", "TV_METAL_THREAD T&")
        code.arg("lr, beta1, beta2, weight_decay, eps, grad_scale", "T")
        code.arg("bias_correction1, bias_correction2_sqrt", "T")
        code.raw(f"""
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        grad /= grad_scale;
        if (weight_decay != 0) {{
            if (IsAdamW) {{
                param -= lr * weight_decay * param;
            }} else {{
                grad += param * weight_decay;
            }}
        }}
        exp_avg = math_op_t::mix(grad, exp_avg, beta1);
        exp_avg_sq = math_op_t::mix(grad * grad, exp_avg_sq, beta2);
        T step_size = lr / bias_correction1;
        T denom = math_op_t::sqrt(exp_avg_sq) / bias_correction2_sqrt + eps;
        param -= step_size * exp_avg / denom;
        """)
        return code

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def adam_correct_zero_grad(self):
        code = pccm.code()
        code.targ("T")
        code.arg("cur_step, target_step", "int")
        # input and output
        code.arg("param, exp_avg, exp_avg_sq", "TV_METAL_THREAD T&")
        code.arg("lr, beta1, beta2, weight_decay, eps", "T")
        code.raw(f"""
        using math_op_t = tv::arrayops::MathScalarOp<T>;
        float beta1_mul = math_op_t::pow(beta1, cur_step);
        float beta2_mul = math_op_t::pow(beta2, cur_step);
        for (int j = cur_step; j < target_step; ++j) {{
            exp_avg *= beta1;
            exp_avg_sq *= beta2;
            float bias_correction1 = 1.0f - beta1_mul;
            float bias_correction2 = 1.0f - beta2_mul;
            float bias_correction2_sqrt = math_op_t::sqrt(bias_correction2);

            T step_size = lr / bias_correction1;
            T denom = math_op_t::sqrt(exp_avg_sq) / bias_correction2_sqrt + eps;
            param -= step_size * exp_avg / denom;

            beta1_mul *= beta1;
            beta2_mul *= beta2;
        }}
        """)
        return code
