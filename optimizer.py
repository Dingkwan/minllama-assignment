from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)


    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每个 param group（可能有不同的 lr / weight_decay 等）
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            correct_bias = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                # 取出当前参数的 state（第一次用需要初始化）
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # step 计数 +1
                state["step"] += 1
                step = state["step"]

                # 一阶矩/二阶矩更新：m_t, v_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # “efficient” bias correction：
                # 直接用校正后的 step_size，而不是显式算 m_hat, v_hat
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                else:
                    step_size = lr

                # 计算 denom = sqrt(v_t) + eps
                denom = exp_avg_sq.sqrt().add_(eps)

                # 参数主更新：theta <- theta - step_size * m_t / (sqrt(v_t) + eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Decoupled weight decay：在梯度更新之后，单独做 L2 衰减
                # 注意：学习率要乘进来
                if weight_decay != 0:
                    p.data.add_(p, alpha=-lr * weight_decay)

        return loss