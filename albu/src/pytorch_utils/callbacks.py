import torch
from copy import deepcopy
import os
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.trainer = None
        self.estimator = None
        self.metrics_collection = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.metrics_collection = trainer.metrics_collection
        self.estimator = trainer.estimator

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            callbacks = callbacks.callbacks
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_batch_end(batch)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class ModelSaver(Callback):
    def __init__(self, save_every, save_name, best_only=True):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name
        self.best_only = best_only

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        need_save = not self.best_only
        if epoch % self.save_every == 0:
            if loss < self.metrics_collection.best_loss:
                self.metrics_collection.best_loss = loss
                self.metrics_collection.best_epoch = epoch
                need_save = True

            if need_save:
                torch.save(deepcopy(self.estimator.model.module),
                           os.path.join(self.estimator.save_path, self.save_name)
                           .format(epoch=epoch, loss="{:.2}".format(loss)))


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
    }, path)


class CheckpointSaver(Callback):
    def __init__(self, save_every, save_name):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        if epoch % self.save_every == 0:
            save_checkpoint(epoch,
                            self.estimator.model.module.state_dict(),
                            self.estimator.optimizer.state_dict(),
                            os.path.join(self.estimator.save_path, self.save_name).format(epoch=epoch, loss="{:.2}".format(loss)))


class LRDropCheckpointSaver(Callback):
    def __init__(self, save_name):
        super().__init__()
        self.save_name = save_name

    def on_epoch_end(self, epoch):
        lr_steps = self.estimator.config.lr_steps
        loss = float(self.metrics_collection.val_metrics['loss'])
        if epoch + 1 in lr_steps:
            save_checkpoint(epoch,
                            self.estimator.model.module.state_dict(),
                            self.estimator.optimizer.state_dict(),
                            os.path.join(self.estimator.save_path, self.save_name).format(epoch=epoch, loss="{:.2}".format(loss)))


class LRStepScheduler(_LRScheduler):
    def __init__(self, optimizer, steps, last_epoch=-1):
        self.lr_steps = steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        pos = max(bisect_right([x for x,y in self.lr_steps], self.last_epoch) - 1, 0)
        return [self.lr_steps[pos][1] if self.lr_steps[pos][0] <= self.last_epoch else base_lr for base_lr in self.base_lrs]



class EarlyStopper(Callback):
    def __init__(self, patience):
        super().__init__()
        self.patience = patience

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        if loss < self.metrics_collection.best_loss:
            self.metrics_collection.best_loss = loss
            self.metrics_collection.best_epoch = epoch
        if epoch - self.metrics_collection.best_epoch >= self.patience:
            self.metrics_collection.stop_training = True

class ModelFreezer(Callback):
    def on_epoch_begin(self, epoch):
        warmup = self.estimator.config.warmup
        if epoch < warmup:
            for p in self.estimator.model.module.encoder_stages.parameters():
                p.requires_grad = False
            if self.estimator.config.num_channels != 3:
                for p in self.estimator.model.module.encoder_stages[0][0].parameters():
                    p.requires_grad = True

            # for param_group in self.estimator.optimizer.param_groups:
            #     param_group['lr'] = 1e-5
        if epoch == warmup:
            for p in self.estimator.model.module.encoder_stages.parameters():
                p.requires_grad = True

            # self.estimator.optimizer = self.estimator.optimizer_type(self.estimator.model.parameters(), lr=self.estimator.config.lr)


class TensorBoard(Callback):
    def __init__(self, logdir):
        super().__init__()
        self.logdir = logdir
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

    def on_epoch_end(self, epoch):
        for k, v in self.metrics_collection.train_metrics.items():
            self.writer.add_scalar('train/{}'.format(k), float(v), global_step=epoch)

        for k, v in self.metrics_collection.val_metrics.items():
            self.writer.add_scalar('val/{}'.format(k), float(v), global_step=epoch)

        for idx, param_group in enumerate(self.estimator.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()

class TelegramSender(Callback):
    def on_train_end(self):
        from telegram_send import send as send_telegram
        message = "Finished on {} with best loss {} on epoch {}".format(
            self.trainer.devices,
            self.trainer.metrics_collection.best_loss or self.metrics_collection.val_metrics['loss'],
            self.trainer.metrics_collection.best_epoch or 'last')
        try:
            send_telegram(messages=message, conf='tg_config.conf')
        except:
            pass
