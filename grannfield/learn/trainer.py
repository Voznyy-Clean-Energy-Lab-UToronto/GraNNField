"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield
"""

import os
import sys
import time

import numpy as np
import torch

from grannfield.learn.meter import AverageMeter
from grannfield.learn.ema import ExponentialMovingAverage


class Trainer:
    r"""Class to train a modules.

    This contains an internal training loop which takes care of validation and can be
    extended with custom functionality using hooks.

    Args:
       model_path (str): path to the modules directory.
       model (torch.Module): modules to be trained.
       loss_fn (callable): training loss function.
       optimizer (torch.optim.optimizer.Optimizer): training optimizer.
       train_loader (torch.utils.data.DataLoader): data loader for training set.
       validation_loader (torch.utils.data.DataLoader): data loader for validation set.
       keep_n_checkpoints (int, optional): number of saved checkpoints.
       checkpoint_interval (int, optional): intervals after which checkpoints is saved.
       hooks (list, optional): hooks to customize training process.
       loss_is_normalized (bool, optional): if True, the loss per data point will be
           reported. Otherwise, the accumulated loss is reported.

    """

    def __init__(
            self,
            model_path,
            model,
            loss_fn,
            optimizer,
            train_loader,
            validation_loader,
            device='cpu',
            clip_grad_norm=None,
            amp=False,
            ema_decay=None,
            normalizers=None,
            print_freq=100,
            keep_n_checkpoints=1000,
            checkpoint_interval=10000,
            validation_interval=10000,
            early_stopping_time=10000,
            early_stopping_lr=1e-6,
            hooks=[],
    ):
        self.model_path = model_path
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        self.best_model = os.path.join(self.model_path, 'best_model.pth.tar')
        self.train_loader = train_loader
        self.print_freq = print_freq
        self.validation_loader = validation_loader
        self.validation_interval = validation_interval
        self.keep_n_checkpoints = keep_n_checkpoints
        self.hooks = hooks
        self.clip_grad_norm = clip_grad_norm
        self.early_stopping_time = early_stopping_time
        self.early_stopping_lr = early_stopping_lr
        self.elapsed = 0
        self.device = device

        self.normalizers = normalizers

        if self.normalizers is not None:
            self.loss_is_normalized = True
        else:
            self.loss_is_normalized = False

        self._model = model.to(self.device)
        self.stop = False
        self.checkpoint_interval = checkpoint_interval

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ema_decay = ema_decay

        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self._model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()
        else:
            os.makedirs(self.checkpoint_path)
            self.step = 0
            self.best_loss = float('inf')
            self.store_checkpoint()

    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    def _optimizer_to(self, device):
        """
        Move the optimizer tensors to device before training.

        Solves restore issue:
        https://github.com/atomistic-machine-learning/schnetpack/issues/126
        https://github.com/pytorch/pytorch/issues/2830

        """
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)


    @property
    def state_dict(self):
        state_dict = {
            'step': self.step,
            'best_loss': self.best_loss,
            'optimizer': self.optimizer.state_dict(),
            'hooks': [h.state_dict for h in self.hooks],
            'normalizers': self.normalizers,
            'ema_decay': self.ema_decay if self.ema else None,
            'ema': self.ema.state_dict() if self.ema else None,
            'amp': self.scaler.state_dict() if self.scaler else None,
            'elapsed': self.elapsed,
        }
        if self._check_is_parallel():
            state_dict['modules'] = self._model.module.state_dict()
        else:
            state_dict['modules'] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.step = state_dict['step']
        self.best_loss = state_dict['best_loss']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._load_model_state_dict(state_dict['modules'])
        self.normalizers = state_dict['normalizers']
        self.ema.load_state_dict(state_dict['ema'])
        self.ema_decay = state_dict['ema_decay']
        self.scaler.load_state_dict(state_dict['scaler'])
        self.elapsed = state_dict['elapsed']

        for h, s in zip(self.hooks, state_dict['hooks']):
            h.state_dict = s

    def store_checkpoint(self, training=True):
        if training:
            chkpt = os.path.join(
                self.checkpoint_path, 'checkpoint-' + str(self.step) + '.pth.tar'
            )
            torch.save(self.state_dict, chkpt)

            chpts = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pth.tar')]
            if len(chpts) > self.keep_n_checkpoints:
                chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
                sidx = np.argsort(chpt_epochs)
                for i in sidx[: -self.keep_n_checkpoints]:
                    os.remove(os.path.join(self.checkpoint_path, chpts[i]))

        else:
            if self.ema:
                self.ema.store()
                self.ema.copy_to()

            chkpt = os.path.join(
                self.checkpoint_path, 'checkpoint-' + str(self.step) + '.pth.tar'
            )
            torch.save(self.state_dict, chkpt)

            chpts = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pth.tar')]
            if len(chpts) > self.keep_n_checkpoints:
                chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
                sidx = np.argsort(chpt_epochs)
                for i in sidx[: -self.keep_n_checkpoints]:
                    os.remove(os.path.join(self.checkpoint_path, chpts[i]))

            if self.ema:
                self.ema.restore()


    def restore_checkpoint(self, path, step=None):
        if step is None:
            step = max(
                [
                    int(f.split('-')[1].split('.')[0])
                    for f in os.listdir(path)
                    if f.startswith('checkpoint')
                ]
            )

        chkpt = os.path.join(
            path, 'checkpoint-' + str(step) + '.pth.tar'
        )
        self.state_dict = torch.load(chkpt)

    def train(self, n_epochs=sys.maxsize):
        """Train the modules for the given number of epochs on a specified device.

        Args:
            device (torch.torch.Device): device on which training takes place.
            n_epochs (int): number of training epochs.

        Note: Depending on the `hooks`, training can stop earlier than `n_epochs`.

        """
        self._optimizer_to(self.device)

        self._stop = False

        start_epoch = self.step // len(self.train_loader)

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            for epoch_int in range(start_epoch, n_epochs):
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()

                epoch_start_time = time.time()

                end = time.time()

                # if self._stop:
                #     # decrease self.epoch if training is aborted on epoch begin
                #     self.epoch -= 1
                #     break

                for h in self.hooks:
                    h.on_epoch_begin(self)

                # perform training epoch
                #                if progress:
                #                    train_iter = tqdm(self.train_loader)
                #                else:
                skip_steps = self.step % len(self.train_loader)
                train_iter = iter(self.train_loader)

                for i in range(skip_steps, len(self.train_loader)):
                    if not self.stop:
                        self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                        self.step = epoch_int * len(self.train_loader) + i + 1
                        self._model.train()

                        train_batch = next(train_iter)

                        data_time.update(time.time() - end)
                        self.optimizer.zero_grad()

                        for h in self.hooks:
                            h.on_batch_begin(self, train_batch)

                        # move input to gpu, if needed
                        for k, v in train_batch.items():
                            train_batch = {k: v.to(self.device) for k, v in train_batch.items()}

                        if self.normalizers is not None:
                            for key in self.normalizers.keys():
                                train_batch[key] = self.normalizers[key].norm(train_batch[key])

                        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                            result = self._model(train_batch)

                        for key in result.keys():
                            if result[key].dim() > train_batch[key].dim():
                                train_batch[key] = train_batch[key].unsqueeze(1)

                        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                            loss = self.loss_fn(train_batch, result)

                        loss = self.scaler.scale(loss) if self.scaler else loss
                        self._backward(loss)
                        scale = self.scaler.get_scale() if self.scaler else 1.0

                        if loss != loss:
                            print('Exit due to NaN')
                            sys.exit(1)

                        if self.normalizers is not None:
                            for key in self.normalizers.keys():
                                result[key] = self.normalizers[key].denorm(result[key])
                                train_batch[key] = self.normalizers[key].denorm(train_batch[key])

                        losses.update(loss.data.cpu() / scale, len(train_batch['idx']))

                        batch_time.update(time.time() - end)
                        end = time.time()

                        if i % self.print_freq == 0:
                            print('[Train Epoch: {0}][{1}/{2}]\t'
                                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
                                epoch_int+1, i+1, len(train_iter), batch_time=batch_time,
                                data_time=data_time, losses=losses))

                        for h in self.hooks:
                            h.on_batch_end(self, train_batch, result, loss)

                        if self.step % self.checkpoint_interval == 0:
                            self.store_checkpoint(training=True)

                        # validation
                    self._model.eval()
                    val_losses = AverageMeter()

                    if self.ema:
                        self.ema.store()
                        self.ema.copy_to()

                    if self.step % self.validation_interval == 0 or self.stop:
                        for h in self.hooks:
                            h.on_validation_begin(self)

                        for i, val_batch in enumerate(self.validation_loader):
                            # append batch_size
                            for h in self.hooks:
                                h.on_validation_batch_begin(self)

                            # move input to gpu, if needed
                            for k, v in val_batch.items():
                                val_batch = {k: v.to(self.device) for k, v in val_batch.items()}

                            if self.normalizers is not None:
                                for key in self.normalizers.keys():
                                    val_batch[key] = self.normalizers[key].norm(val_batch[key])

                            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                                val_result = self._model(val_batch)

                            for key in val_result.keys():
                                if val_result[key].dim() > val_batch[key].dim():
                                    val_batch[key] = val_batch[key].unsqueeze(1)

                            val_batch_loss = (
                                self.loss_fn(val_batch, val_result).data.cpu().numpy()
                            )

                            if self.normalizers is not None:
                                for key in self.normalizers.keys():
                                    val_result[key] = self.normalizers[key].denorm(val_result[key])
                                    val_batch[key] = self.normalizers[key].denorm(val_batch[key])

                            val_losses.update(val_batch_loss, len(val_batch['idx']))

                            batch_time.update(time.time() - end)
                            end = time.time()

                            if i % self.print_freq == 0:
                                print('[Validation][{0}/{1}]\t'
                                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                      'Loss {val_losses.val:.4f} ({val_losses.avg:.4f})'.format(
                                    i+1, len(self.validation_loader), batch_time=batch_time, val_losses=val_losses))

                            for h in self.hooks:
                                h.on_validation_batch_end(self, val_batch, val_result)

                        if self.best_loss > val_losses.avg:
                            self.best_loss = val_losses.avg
                            torch.save(self._model.state_dict(), self.best_model)

                        for h in self.hooks:
                            h.on_validation_end(self, val_losses.avg)

                    if self.ema:
                        self.ema.restore()

                for h in self.hooks:
                    h.on_epoch_end(self)

                torch.cuda.empty_cache()

                if self.early_stopping_time is not None:
                    self.elapsed += time.time() - epoch_start_time
                    if self.elapsed >= self.early_stopping_time:
                        break

                if self._stop:
                    break
            #
            # Training Ends
            #
            # run hooks & store checkpoint
            for h in self.hooks:
                h.on_train_ends(self)

            self.store_checkpoint(training=False)

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)
            raise e

    def test(self, test_loader):

        test_targets = {}
        test_preds = {}

        self._model.eval()

        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        for test_batch in test_loader:
            # append batch_size

            # move input to gpu, if needed
            for k, v in test_batch.items():
                test_batch = {k: v.to(self.device) for k, v in test_batch.items()}

            if self.normalizers is not None:
                for key in self.normalizers.keys():
                    test_batch[key] = self.normalizers[key].norm(test_batch[key])

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                test_result = self._model(test_batch)

            for key in test_result.keys():
                if test_result[key].dim() > test_batch[key].dim():
                    test_batch[key] = test_batch[key].unsqueeze(1)

            test_batch_loss = (
                self.loss_fn(test_batch, test_result).data.cpu().numpy()
            )

            if self.normalizers is not None:
                for key in self.normalizers.keys():
                    test_result[key] = self.normalizers[key].denorm(test_result[key])
                    test_batch[key] = self.normalizers[key].denorm(test_batch[key])

            for key in test_result.keys():
                if key not in test_preds.keys() and key not in test_targets.keys():
                    test_preds[key] = []
                    test_targets[key] = []
                test_preds[key].append(test_result[key])
                test_targets[key].append(test_batch[key])

        if self.ema:
            self.ema.restore()

        return test_preds, test_targets

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # Scale down the gradients of shared parameters
        # if hasattr(self._model.module, "shared_parameters"):
        #     for p, factor in self.model.module.shared_parameters:
        #         if hasattr(p, "grad") and p.grad is not None:
        #             p.grad.detach().div_(factor)
        #         else:
        #             if not hasattr(self, "warned_shared_param_no_grad"):
        #                 self.warned_shared_param_no_grad = True
        #                 print(
        #                     "Some shared parameters do not have a gradient. "
        #                     "Please check if all shared parameters are used "
        #                     "and point to PyTorch parameters."
        #                 )
        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm=self.clip_grad_norm,
            )
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()
