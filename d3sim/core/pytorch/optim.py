from cumm.inliner import NVRTCInlineBuilder
from cumm import dtypes
from typing import Any, Dict, List, Optional, Tuple
import math
from cumm import tensorview as tv
from d3sim.csrc.inliner import INLINER
import pccm
import numpy as np 
import abc 
import torch
from torch.optim import Optimizer
from cumm.inliner import get_current_stream, torch_tensor_to_tv

class NgpOptimizer(abc.ABC):
    @abc.abstractmethod
    def __init__(self,
                 dtype: dtypes.DType,
                 cfg: Dict[str, Any],
                 num_params: int,
                 num_matrix_params: int,
                 external_state: bool = False) -> None:
        ...

    @abc.abstractmethod
    def step(self,
             stream: int,
             loss_scale: float,
             weights_float: tv.Tensor,
             weights: tv.Tensor,
             weights_grad: tv.Tensor,
             state: Optional[Dict[str, tv.Tensor]] = None):
        ...

    @abc.abstractmethod
    def get_current_step(self) -> int:
        ...

    @abc.abstractmethod
    def get_learning_rate(self) -> float:
        ...

    @abc.abstractmethod
    def set_learning_rate(self, val: float):
        ...

    @abc.abstractmethod
    def custom_weights(self) -> Optional[tv.Tensor]:
        ...

    @abc.abstractmethod
    def state_desp(self) -> Dict[str, Tuple[List[int], int]]:
        ...


class NgpAdamRaw(NgpOptimizer):
    def __init__(self,
                 dtype: dtypes.DType,
                 cfg: Dict[str, Any],
                 num_params: int,
                 num_matrix_params: int,
                 external_state: bool = False) -> None:
        self.current_step = 0
        self.dtype = dtype
        self.non_matrix_learning_rate_factor: float = cfg.get(
            "non_matrix_learning_rate_factor", 1.0)
        self.base_learning_rate: float = cfg.get("learning_rate", 1e-3)
        self.beta1: float = cfg.get("beta1", 0.9)
        self.beta2: float = cfg.get("beta2", 0.999)
        self.epsilon: float = cfg.get("epsilon", 1e-8)
        self.l2_reg: float = cfg.get("l2_reg", 1e-8)
        self.relative_weight_decay: float = cfg.get("relative_weight_decay",
                                                    0.0)
        self.absolute_weight_decay: float = cfg.get("absolute_weight_decay",
                                                    0.0)

        self.adabound: bool = cfg.get("adabound", False)
        self.optimize_matrix_params: bool = cfg.get("optimize_matrix_params",
                                                    True)
        self.optimize_non_matrix_params: bool = cfg.get(
            "optimize_non_matrix_params", True)
        self.state: Dict[str, tv.Tensor | torch.Tensor] = {}
        if not external_state:
            self.state = {
                "first_moments": tv.zeros([num_params], tv.float32, 0),
                "second_moments": tv.zeros([num_params], tv.float32, 0),
                "param_steps": tv.zeros([num_params], tv.uint32, 0),
            }
        else:
            self.state = {}
        self.num_params = num_params
        self.num_matrix_params = num_matrix_params

    @classmethod 
    def from_torch_params(cls, params: Dict[str, Any]):
        cfg = {
            "learning_rate": params["lr"],
            "beta1": params["betas"][0],
            "beta2": params["betas"][1],
            "epsilon": params["eps"],
            "l2_reg": params["l2_reg"],
            "relative_weight_decay": params["abs_weight_decay"],
            "absolute_weight_decay": params["rel_weight_decay"],
            "adabound": params["adabound"],
        }
        return cls(dtypes.float32, cfg, 0, 0, True)

    def state_desp(self) -> Dict[str, Tuple[List[int], int]]:
        return {
            "first_moments": ([self.num_params], tv.float32),
            "second_moments": ([self.num_params], tv.float32),
            "param_steps": ([self.num_params], tv.uint32),
        }

    def get_current_step(self):
        return self.current_step

    def get_learning_rate(self):
        return self.base_learning_rate

    def set_learning_rate(self, val: float):
        self.base_learning_rate = val

    def custom_weights(self):
        return None

    def step(self,
             stream: int,
             loss_scale: float,
             weights_float: tv.Tensor | torch.Tensor,
             weights: tv.Tensor | torch.Tensor,
             weights_grad: tv.Tensor | torch.Tensor,
             state: Optional[Dict[str, tv.Tensor | torch.Tensor]] = None,
             weights_grad_inds: Optional[tv.Tensor] = None):
        # if weights_grad_inds is not None, apply sparse optim
        self.current_step += 1
        lower_lr_bound = 0.0
        upper_lr_bound = float(np.finfo(np.float32).max)
        if (self.adabound):
            lower_lr_bound = 0.1 - 0.1 / (
                (1 - self.beta2) * self.current_step + 1)
            upper_lr_bound = 0.1 + 0.1 / ((1 - self.beta2) * self.current_step)
        if state is None:
            state = self.state
        assert state, "state empty"
        first_moments = state["first_moments"]
        second_moments = state["second_moments"]
        if weights_grad_inds is not None:
            assert weights_grad.ndim == 1, "must be 1d tensor"
        param_steps = state["param_steps"]
        code = pccm.code()
        l2_reg = self.l2_reg
        base_learning_rate = self.base_learning_rate
        non_matrix_learning_rate_factor = self.non_matrix_learning_rate_factor
        relative_weight_decay = self.relative_weight_decay
        absolute_weight_decay = self.absolute_weight_decay
        if weights_grad_inds is not None:
            code.raw(f"""
            auto weight_idx = $weights_grad_inds[i];
            """)
        else:
            code.raw(f"""
            auto weight_idx = i;
            """)
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = op::MathScalarOp<float>;
        float gradient = (float)$weights_grad[i] / $loss_scale;
        if (i >= {self.num_matrix_params}) {{
            if (!{pccm.boolean(self.optimize_non_matrix_params)} || gradient == 0.0f) {{
                return;
            }}
        }} else {{
            if (!{pccm.boolean(self.optimize_matrix_params)}) {{
                return;
            }}
        }}
        """)
        code.raw(f"""
        const float weight_fp = $weights_float[weight_idx];

        if (i < {self.num_matrix_params}) {{
            // No L2 reg for non-matrix params
            gradient += $l2_reg * weight_fp;
        }}
        const float gradient_sq = gradient * gradient;

        float first_moment = $first_moments[weight_idx] = {self.beta1}f * $first_moments[weight_idx] + (1.0f - {self.beta1}f) * gradient;
        const float second_moment = $second_moments[weight_idx] = {self.beta2}f * $second_moments[weight_idx] + (1.0f - {self.beta2}f) * gradient_sq;
        float learning_rate = $base_learning_rate;
        if (i >= {self.num_matrix_params}) {{
            // Potentially different learning rate for non-matrix params
            learning_rate *= $non_matrix_learning_rate_factor;
        }}

        // Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
        const uint32_t current_step = ++$param_steps[weight_idx];
        learning_rate *= math_op_t::sqrt(1.0f - math_op_t::pow({self.beta2}f, (float)current_step)) / (1.0f - math_op_t::pow({self.beta1}f, (float)current_step));
        // Follow AdaBound paradigm
        const float effective_learning_rate = math_op_t::clamp(learning_rate / (math_op_t::sqrt(second_moment) + {self.epsilon}f), 
                $lower_lr_bound, $upper_lr_bound);
        auto weight_decay_func = [](float rd, float ad, float weight){{
            return (1.0f - rd) * weight - math_op_t::copysign(ad, weight);
        }};
        const float decayed_weight = weight_decay_func($relative_weight_decay * learning_rate, 
                $absolute_weight_decay * learning_rate, weight_fp);

        const float new_weight = decayed_weight - effective_learning_rate * first_moment;
        $weights_float[weight_idx] = new_weight;
        """)
        # if weights.dtype != tv.float32:
        #     code.raw(f"""
        #     $weights[weight_idx] = std::decay_t<decltype($weights[weight_idx])>(new_weight);
        #     """)
        # param_steps.dtype is int32 in torch wrapped optimizer.
        if weights_grad_inds is not None:
            INLINER.kernel_1d(f"sparse_adam_step_{param_steps.dtype}_{weights.dtype}", weights_grad.numel(), stream, code)
        else:
            INLINER.kernel_1d(f"adam_step_{param_steps.dtype}_{weights.dtype}", weights_grad.numel(), stream, code)




class NgpAdam(Optimizer):
    r"""Implements Ngp Adam algorithm.
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 abs_weight_decay=0,
                 rel_weight_decay=0,
                 adabound: bool = False,
                 l2_reg: float = 1e-8,
                 loss_scale: float = 1.0):
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        abs_weight_decay=abs_weight_decay,
                        rel_weight_decay=rel_weight_decay,
                        adabound=adabound,
                        l2_reg=l2_reg,
                        loss_scale=loss_scale)
        super(NgpAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with tv.measure_and_print("optim"):

            for group in self.param_groups:
                with tv.measure_and_print("optim group prep"):
                    if "__ngp_adam" not in group:
                        ngp_adam = NgpAdamRaw.from_torch_params(group)
                    else:
                        ngp_adam = group["__ngp_adam"]
                    params_with_grad = []
                    grads = []
                    first_moments = []
                    second_moments = []
                    param_steps = []
                    weight_f32: List[Optional[torch.Tensor]] = []
                    for p in group['params']:
                        if p.grad is not None:
                            params_with_grad.append(p)
                            # if p.grad.is_sparse:
                            #     raise RuntimeError(
                            #         'Ngp Adam does not support sparse gradients, please consider SparseAdam instead'
                            #     )
                            grads.append(p.grad)
                            state = self.state[p]
                            # Lazy state initialization
                            if len(state) == 0:
                                state['step'] = 0

                                state['param_steps'] = torch.zeros([p.numel()],
                                                                dtype=torch.int32,
                                                                device=p.device)
                                state['first_moments'] = torch.zeros(
                                    [p.numel()], dtype=torch.float32, device=p.device)
                                state['second_moments'] = torch.zeros(
                                    [p.numel()], dtype=torch.float32, device=p.device)
                                if p.dtype == torch.float16:
                                    # if p is f16, we need to keep a fp32 copy
                                    state['weight_f32'] = torch.zeros(
                                        [p.numel()], dtype=torch.float32, device=p.device)
                                    state['weight_f32'].copy_(p)

                            first_moments.append(state['first_moments'])
                            second_moments.append(state['second_moments'])
                            param_steps.append(state['param_steps'])
                            if "weight_f32" in state:
                                weight_f32.append(state['weight_f32'])
                            else:
                                weight_f32.append(None)
                            state['step'] += 1

                for w, wg, fm, sm, st in zip(params_with_grad, grads,
                                            first_moments, second_moments,
                                            param_steps):
                    # w_tv = torch_tensor_to_tv(w)
                    # fm_tv = torch_tensor_to_tv(fm)
                    # sm_tv = torch_tensor_to_tv(sm)
                    # st_tv = torch_tensor_to_tv(st)
                    state_tv = {
                        "first_moments": fm,
                        "second_moments": sm,
                        "param_steps": st,
                    }
                    # if w32 is not None:
                    #     w32_tv = torch_tensor_to_tv(w32)
                    #     state_tv["weight_f32"] = w32_tv
                    # else:
                    #     w32_tv = w_tv
                    # loss scale is applied externally.
                    if not wg.is_sparse:
                        # with tv.measure_and_print(f"optim step {w.shape}"):
                        ngp_adam.step(get_current_stream(), group["loss_scale"], w, w, wg,
                                    state_tv)
                    else:
                        raise NotImplementedError
                        grad = wg.coalesce()  # the update is non-linear so indices must be unique
                        grad_indices = grad._indices().view(-1)
                        grad_values = grad._values().view(-1)
                        ngp_adam.step(get_current_stream(), group["loss_scale"], w, w, torch_tensor_to_tv(grad_values),
                                    state_tv, torch_tensor_to_tv(grad_indices))


        return loss
