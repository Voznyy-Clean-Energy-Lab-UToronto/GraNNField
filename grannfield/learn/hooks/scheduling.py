"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield

---

This code has been modified from the original version at
atomistic-machine-learning/schnetpack: https://github.com/atomistic-machine-learning/schnetpack
"""

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from grannfield.learn.hooks import Hook


class EarlyStoppingHook(Hook):
    def __init__(self, patience, threshold_ratio=0.0001):

        self.best_loss = float("Inf")
        self.counter = 0
        self.threshold_ratio = threshold_ratio
        self.patience = patience

    @property
    def state_dict(self):
        return {"counter": self.counter, "best_loss": self.best_loss}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.best_loss = state_dict["best_loss"]

    def on_validation_end(self, trainer, val_loss):
        if val_loss > (1 - self.threshold_ratio) * self.best_loss:
            self.counter += 1
        else:
            self.best_loss = val_loss
            self.counter = 0

        if self.counter > self.patience:
            trainer._stop = True


class WarmRestartHook(Hook):
    def __init__(
        self, T0=10, Tmult=2, each_step=False, lr_min=1e-6, lr_factor=1.0, patience=1
    ):
        self.scheduler = None
        self.each_step = each_step
        self.T0 = T0
        self.Tmult = Tmult
        self.Tmax = T0
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.patience = patience
        self.waiting = 0

        self.best_previous = float("Inf")
        self.best_current = float("Inf")

    def on_train_begin(self, trainer):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, self.Tmax, self.lr_min
        )
        self.init_opt_state = trainer.optimizer.state_dict()

    def on_batch_begin(self, trainer, train_batch):
        if self.each_step:
            self.scheduler.step()

    def on_epoch_begin(self, trainer):
        if not self.each_step:
            self.scheduler.step()

    def on_validation_end(self, trainer, val_loss):
        if self.best_current > val_loss:
            self.best_current = val_loss

        if self.scheduler.last_epoch >= self.Tmax:
            self.Tmax *= self.Tmult
            self.scheduler.last_epoch = -1
            self.scheduler.T_max = self.Tmax
            self.scheduler.base_lrs = [
                base_lr * self.lr_factor for base_lr in self.scheduler.base_lrs
            ]
            trainer.optimizer.load_state_dict(self.init_opt_state)

            if self.best_current >= self.best_previous:
                self.waiting += 1
            else:
                self.waiting = 0
                self.best_previous = self.best_current

            if self.waiting > self.patience:
                trainer._stop = True


class MaxEpochHook(Hook):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def on_epoch_begin(self, trainer):
        # stop training if max_epochs is reached
        if trainer.epoch > self.max_epochs:
            trainer._stop = True


class MaxStepHook(Hook):
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def on_batch_begin(self, trainer, train_batch):
        # stop training if max_steps is reached
        if trainer.step > self.max_steps:
            trainer._stop = True


class LRScheduleHook(Hook):
    def __init__(self, scheduler, each_step=False):
        self.scheduler = scheduler
        self.each_step = each_step

    @property
    def state_dict(self):
        return {"scheduler": self.scheduler.state_dict()}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def on_train_begin(self, trainer):
        self.scheduler.last_epoch = trainer.epoch - 1

    def on_batch_begin(self, trainer, train_batch):
        if self.each_step:
            self.scheduler.step()

    def on_epoch_begin(self, trainer):
        if not self.each_step:
            self.scheduler.step()


class ReduceLROnPlateauHook(Hook):
    def __init__(
        self,
        optimizer,
        patience=25,
        factor=0.5,
        min_lr=1e-6,
        window_length=1,
        stop_after_min=False,
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.scheduler = ReduceLROnPlateau(
            optimizer, patience=self.patience, factor=self.factor, min_lr=self.min_lr
        )
        self.window_length = window_length
        self.stop_after_min = stop_after_min
        self.window = []

    @property
    def state_dict(self):
        return {"scheduler": self.scheduler.state_dict()}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def on_validation_end(self, trainer, val_loss):
        self.window.append(val_loss)
        if len(self.window) > self.window_length:
            self.window.pop(0)
        accum_loss = np.mean(self.window)

        self.scheduler.step(accum_loss)

        if self.stop_after_min:
            for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
                old_lr = float(param_group["lr"])
                if old_lr <= self.scheduler.min_lrs[i]:
                    trainer._stop = True


class ExponentialDecayHook(Hook):
    def __init__(self, optimizer, gamma=0.96, step_size=100000):
        self.scheduler = StepLR(optimizer, step_size, gamma)

    def on_batch_end(self, trainer, train_batch, result, loss):
        self.scheduler.step()


class UpdatePrioritiesHook(Hook):
    def __init__(self, prioritized_sampler, priority_fn):
        self.prioritized_sampler = prioritized_sampler
        self.update_fn = priority_fn

    def on_batch_end(self, trainer, train_batch, result, loss):
        idx = train_batch["_idx"]
        self.prioritized_sampler.update_weights(
            idx.data.cpu().squeeze(),
            self.update_fn(train_batch, result).data.cpu().squeeze(),
        )
