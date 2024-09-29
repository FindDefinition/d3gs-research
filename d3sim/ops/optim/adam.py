import math
from numpy import imag
from d3sim.constants import D3SIM_DEFAULT_DEVICE, IsAppleSiliconMacOs
from d3sim.csrc.inliner import INLINER
import torch
import pccm 
from cumm import tensorview as tv 
import dataclasses
from math import sqrt
import torch.nn.functional as F

_ILP = 4

_CHUNK_SIZE = 65536

_BLOCK_SIZE = 512 

STEP_PER_GAUSSIAN = "step_per_gaussian"

@dataclasses.dataclass
class AdamParams:
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0
    eps: float = 1e-8

@dataclasses.dataclass
class AdamState:
    exp_avg: torch.Tensor 
    exp_avg_sq: torch.Tensor 

def fused_adam_single_tensor_step(params: torch.Tensor, grads: torch.Tensor, opt_state: AdamState, opt_params: AdamParams, cur_step: int | torch.Tensor, lr: float, grad_scale: float = 1.0, use_fast_impl: bool = False):
    num = params.numel()

    exp_avg = opt_state.exp_avg
    exp_avg_sq = opt_state.exp_avg_sq

    beta1 = opt_params.beta1 
    beta2 = opt_params.beta2 
    weight_decay = opt_params.weight_decay
    eps = opt_params.eps
    code = pccm.code()
    code.raw(f"""
    using math_op_t = tv::arrayops::MathScalarOp<float>;
    """)

    if not isinstance(cur_step, (int, float)) and not cur_step.is_cpu:
        code.raw(f"""
        auto s = $cur_step[0];
        float bias_correction1_val = 1.0f - math_op_t::pow({beta1}, s);
        float bias_correction2_val = 1.0f - math_op_t::pow({beta2}, s);
        float bias_correction2_sqrt_val = math_op_t::sqrt(bias_correction2_val);
        """)
    else:
        bias_correction1 = 1.0 - math.pow(opt_params.beta1, cur_step)
        bias_correction2 = 1.0 - math.pow(opt_params.beta2, cur_step)
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        code.raw(f"""
        float bias_correction1_val = $bias_correction1;
        float bias_correction2_sqrt_val = $bias_correction2_sqrt;
        """)

    if use_fast_impl:
        # from pytorch adam
        num_chunk = tv.div_up(num, _CHUNK_SIZE)
        num_block_per_chunk = _CHUNK_SIZE // _BLOCK_SIZE

        launch = tv.LaunchParam((num_chunk, 1, 1), (_BLOCK_SIZE, 1, 1))
        code.raw(f"""
        int chunk_idx = tv::parallel::block_idx().x;
        int last_chunk = $num_chunk - 1;
        bool params_aligned = OptimOps::is_aligned_ptr<{_ILP}>($params);
        bool grads_aligned = OptimOps::is_aligned_ptr<{_ILP}>($grads);
        bool exp_avg_aligned = OptimOps::is_aligned_ptr<{_ILP}>($exp_avg);
        bool exp_avg_sq_aligned = OptimOps::is_aligned_ptr<{_ILP}>($exp_avg_sq);

        $params += chunk_idx * {_CHUNK_SIZE};
        $grads += chunk_idx * {_CHUNK_SIZE};
        $exp_avg += chunk_idx * {_CHUNK_SIZE};
        $exp_avg_sq += chunk_idx * {_CHUNK_SIZE};

        tv::alignedarray<float, {_ILP}> params_reg;
        tv::alignedarray<float, {_ILP}> grads_reg;
        tv::alignedarray<float, {_ILP}> exp_avg_reg;
        tv::alignedarray<float, {_ILP}> exp_avg_sq_reg;

        bool all_aligned = params_aligned && grads_aligned && exp_avg_aligned && exp_avg_sq_aligned;
        int remain_work = chunk_idx == last_chunk ? $num - last_chunk * {_CHUNK_SIZE} : {_CHUNK_SIZE};
        if (remain_work % {_ILP} == 0 && all_aligned){{
            for (int j = tv::parallel::thread_idx().x; j * {_ILP} < remain_work; j += {_BLOCK_SIZE}){{
                params_reg = reinterpret_cast<const tv::alignedarray<float, {_ILP}>*>($params)[j];
                grads_reg = reinterpret_cast<const tv::alignedarray<float, {_ILP}>*>($grads)[j];
                exp_avg_reg = reinterpret_cast<const tv::alignedarray<float, {_ILP}>*>($exp_avg)[j];
                exp_avg_sq_reg = reinterpret_cast<const tv::alignedarray<float, {_ILP}>*>($exp_avg_sq)[j];
                TV_PRAGMA_UNROLL
                for (int k = 0; k < {_ILP}; ++k){{
                    OptimOps::adam<float>(grads_reg[k], params_reg[k], exp_avg_reg[k], exp_avg_sq_reg[k],
                        $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
                }}

                reinterpret_cast<tv::alignedarray<float, {_ILP}>*>($params)[j] = params_reg;
                reinterpret_cast<tv::alignedarray<float, {_ILP}>*>($exp_avg)[j] = exp_avg_reg;
                reinterpret_cast<tv::alignedarray<float, {_ILP}>*>($exp_avg_sq)[j] = exp_avg_sq_reg;
            }}
        }}else{{
            for (int j = tv::parallel::thread_idx().x; j < remain_work; j += {_BLOCK_SIZE * _ILP}){{
                TV_PRAGMA_UNROLL
                for (int k = 0; k < {_ILP}; ++k){{
                    int offset = j + k * {_BLOCK_SIZE};
                    bool valid = offset < remain_work;
                    params_reg[k] = valid ? $params[offset] : 0;
                    grads_reg[k] = valid ? $grads[offset] : 0;
                    exp_avg_reg[k] = valid ? $exp_avg[offset] : 0;
                    exp_avg_sq_reg[k] = valid ? $exp_avg_sq[offset] : 0;
                }}
                TV_PRAGMA_UNROLL
                for (int k = 0; k < {_ILP}; ++k){{
                    OptimOps::adam<float>(grads_reg[k], params_reg[k], exp_avg_reg[k], exp_avg_sq_reg[k],
                        $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
                }}
                TV_PRAGMA_UNROLL
                for (int k = 0; k < {_ILP}; ++k){{
                    int offset = j + k * {_BLOCK_SIZE};
                    bool valid = offset < remain_work;
                    if (valid){{
                        $params[offset] = params_reg[k];
                        $exp_avg[offset] = exp_avg_reg[k];
                        $exp_avg_sq[offset] = exp_avg_sq_reg[k];
                    }}
                }}
            }}
        }}
        """)
        INLINER.kernel_raw("fused_adam_single_tensor_fast", launch, code)
    else:
        code.raw(f"""
        auto params_reg = $params[i];
        auto grads_reg = $grads[i];
        auto exp_avg_reg = $exp_avg[i];
        auto exp_avg_sq_reg = $exp_avg_sq[i];
        OptimOps::adam<float>(grads_reg, params_reg, exp_avg_reg, exp_avg_sq_reg,
            $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
        $params[i] = params_reg;
        $exp_avg[i] = exp_avg_reg;
        $exp_avg_sq[i] = exp_avg_sq_reg;
        
        """)
        INLINER.kernel_1d("fused_adam_single_tensor", num, 0, code)

def fused_adam_single_tensor_step_with_mask(mask: torch.Tensor, params: torch.Tensor, grads: torch.Tensor, opt_state: AdamState, opt_params: AdamParams, cur_step: int | torch.Tensor, lr: float, grad_scale: float = 1.0):
    num = params.numel()
    num_mask = mask.numel()
    assert num % num_mask == 0
    num_element_per_mask = num // num_mask
    exp_avg = opt_state.exp_avg
    exp_avg_sq = opt_state.exp_avg_sq
    beta1 = opt_params.beta1 
    beta2 = opt_params.beta2 
    weight_decay = opt_params.weight_decay
    eps = opt_params.eps
    code = pccm.code()
    code.raw(f"""
    using math_op_t = tv::arrayops::MathScalarOp<float>;
    """)

    if not isinstance(cur_step, (int, float)) and not cur_step.is_cpu:
        code.raw(f"""
        auto s = $cur_step[0];
        float bias_correction1_val = 1.0f - math_op_t::pow({beta1}, s);
        float bias_correction2_val = 1.0f - math_op_t::pow({beta2}, s);
        float bias_correction2_sqrt_val = math_op_t::sqrt(bias_correction2_val);
        """)
    else:
        bias_correction1 = 1.0 - math.pow(opt_params.beta1, cur_step)
        bias_correction2 = 1.0 - math.pow(opt_params.beta2, cur_step)
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        code.raw(f"""
        float bias_correction1_val = $bias_correction1;
        float bias_correction2_sqrt_val = $bias_correction2_sqrt;
        """)

    code.raw(f"""
    if ($mask[i / $num_element_per_mask]){{
        auto params_reg = $params[i];
        auto grads_reg = $grads[i];
        auto exp_avg_reg = $exp_avg[i];
        auto exp_avg_sq_reg = $exp_avg_sq[i];
        OptimOps::adam<float>(grads_reg, params_reg, exp_avg_reg, exp_avg_sq_reg,
            $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
        $params[i] = params_reg;
        $exp_avg[i] = exp_avg_reg;
        $exp_avg_sq[i] = exp_avg_sq_reg;
    }}else{{
        auto params_reg = $params[i];
        auto exp_avg_reg = $exp_avg[i];
        auto exp_avg_sq_reg = $exp_avg_sq[i];
        OptimOps::adam<float>(0.0f, params_reg, exp_avg_reg, exp_avg_sq_reg,
            $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
        $params[i] = params_reg;
        $exp_avg[i] = exp_avg_reg;
        $exp_avg_sq[i] = exp_avg_sq_reg;
    }}
    """)
    INLINER.kernel_1d("fused_adam_single_tensor_with_mask", num, 0, code)

def fused_adam_single_tensor_step_with_mask_and_step(mask: torch.Tensor, steps: torch.Tensor, params: torch.Tensor, grads: torch.Tensor, opt_state: AdamState, opt_params: AdamParams, lr: float, grad_scale: float = 1.0):
    num = params.numel()
    num_mask = mask.numel()
    assert num % num_mask == 0 and num_mask == steps.numel()
    num_element_per_mask = num // num_mask
    exp_avg = opt_state.exp_avg
    exp_avg_sq = opt_state.exp_avg_sq
    beta1 = opt_params.beta1 
    beta2 = opt_params.beta2 
    weight_decay = opt_params.weight_decay
    eps = opt_params.eps
    INLINER.kernel_1d("prepare_steps", num_mask, 0, f"""
    if ($mask[i]){{
        ++$steps[i];
    }}
    """)
    code = pccm.code()
    code.raw(f"""
    using math_op_t = tv::arrayops::MathScalarOp<float>;
    int index_per_gauss = i / $num_element_per_mask;
    if ($mask[index_per_gauss]){{
        auto s = $steps[index_per_gauss];
        float bias_correction1_val = 1.0f - math_op_t::pow($beta1, s);
        float bias_correction2_val = 1.0f - math_op_t::pow($beta2, s);
        float bias_correction2_sqrt_val = math_op_t::sqrt(bias_correction2_val);

        auto params_reg = $params[i];
        auto grads_reg = $grads[i];
        auto exp_avg_reg = $exp_avg[i];
        auto exp_avg_sq_reg = $exp_avg_sq[i];
        OptimOps::adam<float>(grads_reg, params_reg, exp_avg_reg, exp_avg_sq_reg,
            $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
        $params[i] = params_reg;
        $exp_avg[i] = exp_avg_reg;
        $exp_avg_sq[i] = exp_avg_sq_reg;
    }}
    """)
    INLINER.kernel_1d("fused_adam_single_tensor_with_mask_and_step", num, 0, code)

def fused_adam_single_tensor_step_with_mask_and_step_lossless(mask: torch.Tensor, steps: torch.Tensor, params: torch.Tensor, grads: torch.Tensor, opt_state: AdamState, opt_params: AdamParams, cur_step: int | torch.Tensor, lr: float, grad_scale: float = 1.0):
    num = params.numel()
    num_mask = mask.numel()
    assert num % num_mask == 0 and num_mask == steps.numel()
    num_element_per_mask = num // num_mask
    exp_avg = opt_state.exp_avg
    exp_avg_sq = opt_state.exp_avg_sq
    beta1 = opt_params.beta1 
    beta2 = opt_params.beta2 
    weight_decay = opt_params.weight_decay
    assert weight_decay == 0, "weight decay is not supported when use lossless sparse adam"
    eps = opt_params.eps
    code = pccm.code()
    if not isinstance(cur_step, (int, float)) and not cur_step.is_cpu:
        code.raw(f"""
        int s_target = $cur_step[0];
        """)
    else:
        step_cpu_val = int(cur_step)
        code.raw(f"""
        int s_target = $step_cpu_val;
        """)
    """
        for (int j = s; j < s_target; ++j){{
            float bias_correction1_val = 1.0f - math_op_t::pow($beta1, j);
            float bias_correction2_val = 1.0f - math_op_t::pow($beta2, j);
            float bias_correction2_sqrt_val = math_op_t::sqrt(bias_correction2_val);
            OptimOps::adam<float>(0.0f, params_reg, exp_avg_reg, exp_avg_sq_reg,
                $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
        }}

    """
    code.raw(f"""
    using math_op_t = tv::arrayops::MathScalarOp<float>;
    int index_per_gauss = i / $num_element_per_mask;
    if ($mask[index_per_gauss]){{

        auto params_reg = $params[i];
        auto grads_reg = $grads[i];
        auto exp_avg_reg = $exp_avg[i];
        auto exp_avg_sq_reg = $exp_avg_sq[i];

        int s = $steps[index_per_gauss] + 1;
        if (exp_avg_reg != 0 && exp_avg_sq_reg != 0){{
            OptimOps::adam_correct_zero_grad<float>(s, s_target, params_reg, exp_avg_reg, exp_avg_sq_reg,
                    $lr, $beta1, $beta2, $weight_decay, $eps);
        }}
        float bias_correction1_val = 1.0f - math_op_t::pow($beta1, s_target);
        float bias_correction2_val = 1.0f - math_op_t::pow($beta2, s_target);
        float bias_correction2_sqrt_val = math_op_t::sqrt(bias_correction2_val);
        OptimOps::adam<float>(grads_reg, params_reg, exp_avg_reg, exp_avg_sq_reg,
            $lr, $beta1, $beta2, $weight_decay, $eps, $grad_scale, bias_correction1_val, bias_correction2_sqrt_val);
        $params[i] = params_reg;
        $exp_avg[i] = exp_avg_reg;
        $exp_avg_sq[i] = exp_avg_sq_reg;
    }}
    """)
    INLINER.kernel_1d("fused_adam_single_tensor_with_mask_and_step", num, 0, code)
    code = pccm.code()
    if not isinstance(cur_step, (int, float)) and not cur_step.is_cpu:
        code.raw(f"""
        int s_target = $cur_step[0];
        """)
    else:
        step_cpu_val = int(cur_step)
        code.raw(f"""
        int s_target = $step_cpu_val;
        """)
    code.raw(f"""
    if ($mask[i]){{
        $steps[i] = s_target;
    }}
    """)
    
    INLINER.kernel_1d("prepare_steps", num_mask, 0, code)

def fused_adam_with_step_lossless_update(steps: torch.Tensor, params: torch.Tensor, grads: torch.Tensor, opt_state: AdamState, opt_params: AdamParams, cur_step: int | torch.Tensor, lr: float, grad_scale: float = 1.0):
    num = params.numel()
    num_mask = steps.numel()
    assert num % num_mask == 0
    num_element_per_mask = num // num_mask
    exp_avg = opt_state.exp_avg
    exp_avg_sq = opt_state.exp_avg_sq
    beta1 = opt_params.beta1 
    beta2 = opt_params.beta2 
    weight_decay = opt_params.weight_decay
    assert weight_decay == 0, "weight decay is not supported when use lossless sparse adam"
    eps = opt_params.eps
    code = pccm.code()
    if not isinstance(cur_step, (int, float)) and not cur_step.is_cpu:
        code.raw(f"""
        int s_target = $cur_step[0];
        """)
    else:
        step_cpu_val = int(cur_step)
        code.raw(f"""
        int s_target = $step_cpu_val;
        """)
    code.raw(f"""
    using math_op_t = tv::arrayops::MathScalarOp<float>;
    int index_per_gauss = i / $num_element_per_mask;
    int s = $steps[index_per_gauss];
    s += 1;
    s_target += 1;
    if (s != s_target){{

        auto params_reg = $params[i];
        auto grads_reg = $grads[i];
        auto exp_avg_reg = $exp_avg[i];
        auto exp_avg_sq_reg = $exp_avg_sq[i];

        if (exp_avg_reg != 0 && exp_avg_sq_reg != 0){{
            OptimOps::adam_correct_zero_grad<float>(s, s_target, params_reg, exp_avg_reg, exp_avg_sq_reg,
                    $lr, $beta1, $beta2, $weight_decay, $eps);
            $params[i] = params_reg;
            $exp_avg[i] = exp_avg_reg;
            $exp_avg_sq[i] = exp_avg_sq_reg;
        }}
    }}
    """)
    INLINER.kernel_1d("fused_adam_with_step_lossless_update", num, 0, code)
    code = pccm.code()
    if not isinstance(cur_step, (int, float)) and not cur_step.is_cpu:
        code.raw(f"""
        int s_target = $cur_step[0];
        """)
    else:
        step_cpu_val = int(cur_step)
        code.raw(f"""
        int s_target = $step_cpu_val;
        """)
    code.raw(f"""
    $steps[i] = s_target;
    """)
    
    INLINER.kernel_1d("prepare_steps_lossless_update", num_mask, 0, code)

class GaussianAdam(torch.optim.Adam):
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            fused = group["fused"]
            assert not fused
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            beta1, beta2 = group["betas"]
            adam_params = AdamParams(beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"])
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            for j in range(len(params_with_grad)):
                state_steps[j] += 1
                fused_adam_single_tensor_step(params_with_grad[j], grads[j], AdamState(exp_avgs[j], exp_avg_sqs[j]), adam_params, state_steps[j], group["lr"], use_fast_impl=True)
        return loss

    def step_with_mask(self, mask: torch.Tensor, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            fused = group["fused"]
            assert not fused
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            beta1, beta2 = group["betas"]
            adam_params = AdamParams(beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"])
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            for j in range(len(params_with_grad)):
                state_steps[j] += 1
                fused_adam_single_tensor_step_with_mask(mask, params_with_grad[j], grads[j], AdamState(exp_avgs[j], exp_avg_sqs[j]), adam_params, state_steps[j], group["lr"])
        return loss

class GaussianSparseAdam(torch.optim.Adam):
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        raise NotImplementedError("use step_with_mask")

    def sparse_step_update(self):
        self._cuda_graph_capture_health_check()
        loss = None
        for group in self.param_groups:
            fused = group["fused"]
            assert not fused
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            step_per_gaussians: list[torch.Tensor] = []
            beta1, beta2 = group["betas"]
            adam_params = AdamParams(beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"])
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            for p in params_with_grad:
                state = self.state[p]
                step_per_gaussians.append(state[STEP_PER_GAUSSIAN])
            for j in range(len(params_with_grad)):
                fused_adam_with_step_lossless_update(step_per_gaussians[j], params_with_grad[j], grads[j], AdamState(exp_avgs[j], exp_avg_sqs[j]), adam_params, state_steps[j], group["lr"])
        return

    def step_with_mask(self, mask: torch.Tensor, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            fused = group["fused"]
            assert not fused
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            step_per_gaussians: list[torch.Tensor] = []
            beta1, beta2 = group["betas"]
            adam_params = AdamParams(beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"])
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            for p in params_with_grad:
                state = self.state[p]
                if STEP_PER_GAUSSIAN not in state:
                    state[STEP_PER_GAUSSIAN] = torch.zeros([p.shape[0]], device=p.device, dtype=torch.int32)
                step_per_gaussians.append(state[STEP_PER_GAUSSIAN])
            for j in range(len(params_with_grad)):
                state_steps[j] += 1
                fused_adam_single_tensor_step_with_mask_and_step_lossless(mask, step_per_gaussians[j], params_with_grad[j], grads[j], AdamState(exp_avgs[j], exp_avg_sqs[j]), adam_params, state_steps[j], group["lr"])
        return loss


def _test_optim_step():
    params = torch.randn(50000000, device=D3SIM_DEFAULT_DEVICE, requires_grad=True)
    params = torch.nn.Parameter(params)
    lr = 0.01
    optim_ref = torch.optim.Adam([params], lr=lr, fused=True)
    grads = torch.randn(50000000, device=D3SIM_DEFAULT_DEVICE)

    for j in range(1):
        optim_ref.zero_grad()
        params.grad = grads.clone()
        optim_ref.step()
    for j in range(2):
        cur_grads = grads.clone()
        # get optim state
        p = optim_ref.param_groups[0]["params"][0]
        step = optim_ref.state[p]["step"]
        step_int = float(step.item()) + 1
        exp_avg = optim_ref.state[p]["exp_avg"]
        exp_avg_sq = optim_ref.state[p]["exp_avg_sq"]
        params_to_test = params.detach().clone()
        exp_avg_to_test = exp_avg.detach().clone()
        exp_avg_sq_to_test = exp_avg_sq.detach().clone()
        print(type(step), step_int, step.device, step.dtype)
        opt_state = AdamState(
            exp_avg=exp_avg_to_test,
            exp_avg_sq=exp_avg_sq_to_test
        )
        opt_params = AdamParams()
        cur_step = step
        with tv.measure_and_print("adam my"):
            fused_adam_single_tensor_step(params_to_test, cur_grads, opt_state, opt_params, step_int, lr, use_fast_impl=True)


        # ref step
        optim_ref.zero_grad()
        params.grad = cur_grads.clone()
        with tv.measure_and_print("adam ref"):

            optim_ref.step()
        print(torch.linalg.norm(params_to_test - params))
        print(torch.linalg.norm(exp_avg_to_test - exp_avg))
        print(torch.linalg.norm(exp_avg_sq_to_test - exp_avg_sq))

    # breakpoint()
    # print("?")

def _test_optim_step_v2():
    params = torch.randn(50000000, device=D3SIM_DEFAULT_DEVICE, requires_grad=True)
    params = torch.nn.Parameter(params)
    lr = 0.01
    optim_ref = torch.optim.Adam([params], lr=lr, fused=True)
    params_my = torch.nn.Parameter(params.clone().requires_grad_(True))
    optim_my = GaussianAdam([params_my], lr=lr)

    grads = torch.randn(50000000, device=D3SIM_DEFAULT_DEVICE)
    for j in range(2):
        cur_grads = grads.clone()
        if j > 0:
            p = optim_ref.param_groups[0]["params"][0]
            exp_avg_ref = optim_ref.state[p]["exp_avg"]
            exp_avg_sq_ref = optim_ref.state[p]["exp_avg_sq"]

            p_my = optim_my.param_groups[0]["params"][0]
            exp_avg_my = optim_my.state[p_my]["exp_avg"]
            exp_avg_sq_my = optim_my.state[p_my]["exp_avg_sq"]
            p_my.data.copy_(p.data)
            exp_avg_my.copy_(exp_avg_ref)
            exp_avg_sq_my.copy_(exp_avg_sq_ref)

        # ref step
        optim_ref.zero_grad()
        params.grad = cur_grads.clone()
        with tv.measure_and_print("adam ref"):

            optim_ref.step()
        optim_my.zero_grad()
        params_my.grad = cur_grads.clone()
        with tv.measure_and_print("adam my"):
            optim_my.step()

        p = optim_ref.param_groups[0]["params"][0]
        exp_avg_ref = optim_ref.state[p]["exp_avg"]
        exp_avg_sq_ref = optim_ref.state[p]["exp_avg_sq"]

        p = optim_my.param_groups[0]["params"][0]
        exp_avg_my = optim_my.state[p]["exp_avg"]
        exp_avg_sq_my = optim_my.state[p]["exp_avg_sq"]

        print(torch.linalg.norm(params_my - params))
        print(torch.linalg.norm(exp_avg_my - exp_avg_ref))
        print(torch.linalg.norm(exp_avg_sq_my - exp_avg_sq_ref))

    # breakpoint()
    # print("?")

def _test_optim_step_sparse_gs_adam():
    params = torch.randn(5000000, 10, device=D3SIM_DEFAULT_DEVICE, requires_grad=True)
    params = torch.nn.Parameter(params)
    lr = 0.01
    optim_ref = torch.optim.Adam([params], lr=lr, fused=True)
    params_my = torch.nn.Parameter(params.clone().requires_grad_(True))
    optim_my = GaussianSparseAdam([params_my], lr=lr)

    grads = torch.randn(5000000, 10, device=D3SIM_DEFAULT_DEVICE)
    for j in range(10):
        random_mask = torch.randint(0, 2, [params.shape[0]], device=D3SIM_DEFAULT_DEVICE, dtype=torch.uint8)
        if j == 9 or j % 3 == 0:
            random_mask[:] = 1
        cur_grads = grads.clone()
        cur_grads[random_mask == 0] = 0
        # ref step
        optim_ref.zero_grad()
        params.grad = cur_grads.clone()
        with tv.measure_and_print("adam ref"):

            optim_ref.step()
        optim_my.zero_grad()
        params_my.grad = cur_grads.clone()
        with tv.measure_and_print("adam my"):
            optim_my.step_with_mask(random_mask)
        optim_my.sparse_step_update()
        p = optim_ref.param_groups[0]["params"][0]
        exp_avg_ref = optim_ref.state[p]["exp_avg"]
        exp_avg_sq_ref = optim_ref.state[p]["exp_avg_sq"]

        p = optim_my.param_groups[0]["params"][0]
        exp_avg_my = optim_my.state[p]["exp_avg"]
        exp_avg_sq_my = optim_my.state[p]["exp_avg_sq"]

        print(torch.linalg.norm(params_my - params))
        print(torch.linalg.norm(exp_avg_my - exp_avg_ref))
        print(torch.linalg.norm(exp_avg_sq_my - exp_avg_sq_ref))



if __name__ == "__main__":
    with torch.no_grad():
        _test_optim_step_sparse_gs_adam()   