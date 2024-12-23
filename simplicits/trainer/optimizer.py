"""
Some bits taken from https://github.com/facebookresearch/schedule_free.
"""

import os

import torch
import torch.distributed as dist


@torch.compile
def oth(weights):
    if weights.size(0) < weights.size(1):
        weights = weights.T
        transposed = True
    else:
        transposed = False

    Q, R = torch.linalg.qr(weights)
    d = torch.diag(R)
    ph = d.sign()
    Q *= ph.unsqueeze(0)

    if transposed:
        Q = Q.T

    return Q.contiguous().type_as(weights)


class DistributedSimplicitsOptim(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, ns_steps=3):
        defaults = dict(lr=lr, betas=betas, eps=eps, ns_steps=ns_steps)
        super().__init__(params, defaults)

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def _init_state(self, p):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, device=p.device)
        state["exp_avg_sq"] = torch.zeros_like(p, device=p.device)
        return state

    @torch.compile()
    def _stable_ns(self, X, steps=3):
        a, b, c = (3.4445, -4.7750, 2.0315)

        X = X.bfloat16()

        norm = X.norm() + 1e-7
        X = X / norm

        transposed = False
        if X.size(0) < X.size(1):
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

            if torch.isnan(X).any() or torch.isinf(X).any():
                X = self._safe_normalize(X)

        if transposed:
            X = X.T

        return X.to(dtype=torch.float32).contiguous()

    def _safe_normalize(self, X):
        norm = torch.clamp(X.norm(), min=1e-7)
        return X / norm

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if len(self.state[p]) == 0:
                    self._init_state(p)

                state = self.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                ns_steps = group["ns_steps"]

                state["step"] += 1
                grad = p.grad

                torch.nn.utils.clip_grad_norm_(p, 1.0)

                # Adam-like update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * (bias_correction2**0.5) / bias_correction1

                p.data.addcdiv_(
                    exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size
                )

            params_2d = [
                p for p in group["params"] if p.grad is not None and len(p.shape) == 2
            ]
            if params_2d:
                total_params = sum(p.numel() for p in params_2d)
                updates_flat = torch.zeros(
                    total_params, device="cuda", dtype=torch.bfloat16
                )

                curr_idx = 0
                for i, p in enumerate(params_2d):
                    if i % self.world_size == self.rank:
                        update = self._stable_ns(p.data, steps=ns_steps)
                        updates_flat[curr_idx : curr_idx + p.numel()] = update.flatten()
                    curr_idx += p.numel()

                if self.world_size > 1:
                    dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

                curr_idx = 0
                for p in params_2d:
                    update = updates_flat[curr_idx : curr_idx + p.numel()].view_as(
                        p.data
                    )
                    p.data.copy_(update.to(p.dtype))
                    curr_idx += p.numel()


class SimplicitsOptim(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, ns_steps=3):
        defaults = dict(lr=lr, betas=betas, eps=eps, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def _init_state(self, p):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, device=p.device)
        state["exp_avg_sq"] = torch.zeros_like(p, device=p.device)
        return state

    @torch.compile()
    def _stable_ns(self, X, steps=3):
        a, b, c = (3.4445, -4.7750, 2.0315)

        X = X.bfloat16()

        norm = X.norm() + 1e-7
        X = X / norm

        transposed = False
        if X.size(0) < X.size(1):
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

            if torch.isnan(X).any() or torch.isinf(X).any():
                X = self._safe_normalize(X)

        if transposed:
            X = X.T

        return X.to(dtype=torch.float32).contiguous()

    def _safe_normalize(self, X):
        norm = torch.clamp(X.norm(), min=1e-7)
        return X / norm

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if len(self.state[p]) == 0:
                    self._init_state(p)

                state = self.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                ns_steps = group["ns_steps"]

                state["step"] += 1
                grad = p.grad

                torch.nn.utils.clip_grad_norm_(p, 1.0)

                # Adam-like update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * (bias_correction2**0.5) / bias_correction1

                p.data.addcdiv_(
                    exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size
                )

                if len(p.shape) == 2:
                    p.data = self._stable_ns(p.data, steps=ns_steps)
