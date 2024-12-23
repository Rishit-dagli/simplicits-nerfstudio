from typing import List

import torch


class LinearLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    torch has a linear scheduler however it works a bit differently than what i
    wanted, this class makes it a bit easier to use it.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_lr: float,
        end_lr: float,
        num_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_steps = num_steps
        super(LinearLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < 0:
            return [self.start_lr for _ in self.optimizer.param_groups]
        elif self.last_epoch >= self.num_steps:
            return [self.end_lr for _ in self.optimizer.param_groups]
        else:
            progress = self.last_epoch / self.num_steps
            lr = self.start_lr + progress * (self.end_lr - self.start_lr)
            return [lr for _ in self.optimizer.param_groups]
