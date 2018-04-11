import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from tqdm import tqdm
from typing import Type

from dataset.neural_dataset import TrainDataset, ValDataset
from .loss import dice_round, dice_loss, multi_class_dice, multi_class_dice_round, jaccard, jaccard_round
from .callbacks import EarlyStopper, ModelSaver, TensorBoard, CheckpointSaver, Callbacks, LRDropCheckpointSaver, ModelFreezer, LRStepScheduler
from pytorch_zoo import unet, resnet38unet

models = {
    'resnet34_upsample': unet.Resnet34_upsample,
    'resnet34_sum': unet.Resnet34_sum,
    'resnet34_double': unet.Resnet34_double,
    'resnet34_bn': unet.Resnet34_bn_sum,
    'resnet34_dil': unet.DilatedResnet34,
    'dpn': unet.DPNUnet,
    'incv3': unet.Incv3,
    'resnet38': resnet38unet.WideResnet38,
    'vgg11bn': unet.Vgg11bn,
    'vgg16bn': unet.Vgg16bn,
}

optimizers = {
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}

class Estimator:
    def __init__(self, model: torch.nn.Module, optimizer: Type[optim.Optimizer], save_path,
                 config, num_channels_changed=False, final_changed=False):
        self.model = nn.DataParallel(model).cuda()
        self.optimizer = optimizer(self.model.parameters(), lr=config.lr)
        self.start_epoch = 0
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.num_channels_changed = num_channels_changed
        self.final_changed = final_changed
        self.iter_size = config.iter_size

        self.lr_scheduler = None
        self.lr = config.lr
        self.config = config
        self.optimizer_type = optimizer

    def resume(self, checkpoint_name):
        try:
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint_name))
        except FileNotFoundError:
            print("resume failed, file not found")
            return False

        self.start_epoch = checkpoint['epoch']

        model_dict = self.model.module.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if self.num_channels_changed or self.final_changed:
            skip_layers = self.model.module.first_layer_params_names if self.num_channels_changed else self.model.module.last_layer_params_names
            print('skipping: ', [k for k in pretrained_dict.keys() if any(s in k for s in skip_layers)])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not any(s in k for s in skip_layers)}
            model_dict.update(pretrained_dict)
            self.model.module.load_state_dict(model_dict)
        else:
            model_dict.update(pretrained_dict)
            self.model.module.load_state_dict(model_dict)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        print("resumed from checkpoint {} on epoch: {}".format(os.path.join(self.save_path, checkpoint_name), self.start_epoch))
        return True

    def calculate_loss_multichannel(self, output, target, meter, training, iter_size):
        bce = F.binary_cross_entropy_with_logits(output, target)
        output = F.sigmoid(output)
        dice = dice_loss(output, target)
        dice_ch2 = dice_loss(output[:,-1,...], target[:,-1,...])
        dice_r = dice_round(output[:,-1,...], target[:,-1,...])

        loss = (self.config.loss['bce'] * bce + self.config.loss['dice'] * (1 - dice)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['dice'] += dice_ch2.data.cpu().numpy()[0] / iter_size
        meter['bce'] += bce.data.cpu().numpy()[0] / iter_size
        meter['dr'] += dice_r.data.cpu().numpy()[0] / iter_size
        return meter

    def calculate_loss_single_channel(self, output, target, meter, training, iter_size):
        bce = nn.BCEWithLogitsLoss()(output, target)
        output = F.sigmoid(output)
        dice = dice_loss(output, target)
        dice_r = dice_round(output, target)

        loss = (self.config.loss['bce'] * bce + self.config.loss['dice'] * (1 - dice)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['dice'] += dice.data.cpu().numpy()[0] / iter_size
        meter['bce'] += bce.data.cpu().numpy()[0] / iter_size
        meter['dr'] += dice_r.data.cpu().numpy()[0] / iter_size
        return meter

    def calculate_loss_softmax(self, output, target, meter, training, iter_size):
        ce = F.cross_entropy(output, target)
        output = F.softmax(output, dim=1)
        dice_body = dice_loss(output[:,2,...], (target==2).float())
        dice_border = dice_loss(output[:,1,...], (target==1).float())
        dice_r_body = dice_round(output[:,2,...], (target==2).float())
        dice_r_border = dice_round(output[:,1,...], (target==1).float())

        loss = (self.config.loss['ce'] * ce + self.config.loss['dice_body'] * (1 - dice_body) + self.config.loss['dice_border'] * (1 - dice_border)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['d_n'] += dice_body.data.cpu().numpy()[0] / iter_size
        meter['d_b'] += dice_border.data.cpu().numpy()[0] / iter_size
        meter['ce'] += ce.data.cpu().numpy()[0] / iter_size
        meter['dr_n'] += dice_r_body.data.cpu().numpy()[0] / iter_size
        meter['dr_b'] += dice_r_border.data.cpu().numpy()[0] / iter_size
        return meter

    def calculate_loss_sigmoid(self, output, target, meter, training, iter_size):
        ce_body = F.binary_cross_entropy_with_logits(output[:,0,...], target[:,2,...])
        ce_border = F.binary_cross_entropy_with_logits(output[:,1,...], target[:,1,...])
        ce = ce_body + ce_border
        output = F.sigmoid(output)
        dice_body = dice_loss(output[:,0,...], target[:,2,...])
        dice_border = dice_loss(output[:,1,...], target[:,1,...])
        dice_r_body = dice_round(output[:,0,...], target[:,2,...])
        dice_r_border = dice_round(output[:,1,...], target[:,1,...])

        loss = (self.config.loss['ce'] * ce + self.config.loss['dice_body'] * (1 - dice_body) + self.config.loss['dice_border'] * (1 - dice_border)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['d_n'] += dice_body.data.cpu().numpy()[0] / iter_size
        meter['d_b'] += dice_border.data.cpu().numpy()[0] / iter_size
        meter['ce'] += ce.data.cpu().numpy()[0] / iter_size
        meter['dr_n'] += dice_r_body.data.cpu().numpy()[0] / iter_size
        meter['dr_b'] += dice_r_border.data.cpu().numpy()[0] / iter_size
        return meter

    def calculate_loss_3ch(self, output, target, meter, training, iter_size):
        ce_body = (F.binary_cross_entropy_with_logits(output[:,2,...], target[:,2,...]) +
                   F.binary_cross_entropy_with_logits(output[:,0,...], target[:,0,...])) / 2
        ce_border = F.binary_cross_entropy_with_logits(output[:,1,...], target[:,1,...])
        ce = ce_body + ce_border
        output = F.sigmoid(output)
        dice_body = (dice_loss(output[:,0,...], target[:,0,...]) + dice_loss(output[:,2,...], target[:,2,...])) / 2
        dice_border = dice_loss(output[:,1,...], target[:,1,...])
        dice_r_body = (dice_round(output[:,0,...], target[:,0,...]) + dice_round(output[:,2,...], target[:,2,...])) / 2
        dice_r_border = dice_round(output[:,1,...], target[:,1,...])

        loss = (self.config.loss['ce'] * ce + self.config.loss['dice_body'] * (1 - dice_body) + self.config.loss['dice_border'] * (1 - dice_border)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['d_n'] += dice_body.data.cpu().numpy()[0] / iter_size
        meter['d_b'] += dice_border.data.cpu().numpy()[0] / iter_size
        meter['ce'] += ce.data.cpu().numpy()[0] / iter_size
        meter['dr_n'] += dice_r_body.data.cpu().numpy()[0] / iter_size
        meter['dr_b'] += dice_r_border.data.cpu().numpy()[0] / iter_size
        return meter


    def make_step_itersize(self, images, ytrues, training):
        iter_size = self.iter_size
        if training:
            self.optimizer.zero_grad()

        inputs = images.chunk(iter_size)
        targets = ytrues.chunk(iter_size)

        meter = defaultdict(float)
        for input, target in zip(inputs, targets):
            input = torch.autograd.Variable(input.cuda(async=True), volatile=not training)
            target = torch.autograd.Variable(target.cuda(async=True), volatile=not training)
            output = self.model(input)
            if self.config.sigmoid:
                meter = self.calculate_loss_3ch(output, target, meter, training, iter_size)
            else:
                meter = self.calculate_loss_softmax(output, target, meter, training, iter_size)
            # meter = self.calculate_loss_single_channel(output, target, meter, training, iter_size)
            #additional metrics
            # for name, func in metrics:
            #     acc = func(output.contiguous(), target.contiguous())
            #     meter[name] += acc.data.cpu().numpy()[0] / iter_size

            # outputs.append(output.data)

        if training:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
            self.optimizer.step()
        return meter, None#torch.cat(outputs, dim=0)

class MetricsCollection:
    def __init__(self):
        self.stop_training = False
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class PytorchTrain:
    def __init__(self, estimator: Estimator, fold, callbacks=None, hard_negative_miner=None):
        self.fold = fold
        self.estimator = estimator

        self.devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        if os.name == 'nt':
            self.devices = ','.join(str(d + 5) for d in map(int, self.devices.split(',')))

        self.hard_negative_miner = hard_negative_miner
        self.metrics_collection = MetricsCollection()

        self.estimator.resume("fold" + str(fold) + "_checkpoint.pth")
        # if self.estimator.model_changed:
        #     callbacks.append(ColdStart(self.estimator.lr, 5, 30, 0.1))

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def _run_one_epoch(self, epoch, loader, training=True):
        avg_meter = defaultdict(float)

        pbar = tqdm(enumerate(loader), total=len(loader), desc="Epoch {}{}".format(epoch, ' eval' if not training else ""), ncols=0)
        for i, data in pbar:
            self.callbacks.on_batch_begin(i)

            meter, ypreds = self._make_step(data, training)
            for k, val in meter.items():
                avg_meter[k] += val

            if training:
                if self.hard_negative_miner is not None:
                    self.hard_negative_miner.update_cache(meter, data)
                    if self.hard_negative_miner.need_iter():
                        self._make_step(self.hard_negative_miner.cache, training)
                        self.hard_negative_miner.invalidate_cache()

            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})

            self.callbacks.on_batch_end(i)
        return {k: v / len(loader) for k, v in avg_meter.items()}

    def _make_step(self, data, training):
        images = data['image']
        ytrues = data['mask']

        meter, ypreds = self.estimator.make_step_itersize(images, ytrues, training)

        return meter, ypreds

    def fit(self, train_loader, val_loader, nb_epoch):
        self.callbacks.on_train_begin()

        for epoch in range(self.estimator.start_epoch, nb_epoch):
            self.callbacks.on_epoch_begin(epoch)

            if self.estimator.lr_scheduler is not None and epoch >= self.estimator.config.warmup:
                self.estimator.lr_scheduler.step(epoch)

            self.estimator.model.train()
            self.metrics_collection.train_metrics = self._run_one_epoch(epoch, train_loader, training=True)
            self.estimator.model.eval()
            self.metrics_collection.val_metrics = self._run_one_epoch(epoch, val_loader, training=False)

            self.callbacks.on_epoch_end(epoch)

            if self.metrics_collection.stop_training:
                break

        self.callbacks.on_train_end()


def train(ds, val_ds, fold, train_idx, val_idx, config, num_workers=0, transforms=None, val_transforms=None, num_channels_changed=False, final_changed=False, cycle=False):
    os.makedirs(os.path.join('..', 'weights'), exist_ok=True)
    os.makedirs(os.path.join('..', 'logs'), exist_ok=True)

    save_path = os.path.join('..', 'weights', config.folder)
    model = models[config.network](num_classes=config.num_classes, num_channels=config.num_channels)
    estimator = Estimator(model, optimizers[config.optimizer], save_path,
                          config=config, num_channels_changed=num_channels_changed, final_changed=final_changed)

    estimator.lr_scheduler = ExponentialLR(estimator.optimizer, config.lr_gamma)#LRStepScheduler(estimator.optimizer, config.lr_steps)
    callbacks = [
        ModelSaver(1, ("fold"+str(fold)+"_best.pth"), best_only=True),
        ModelSaver(1, ("fold"+str(fold)+"_last.pth"), best_only=False),
        CheckpointSaver(1, ("fold"+str(fold)+"_checkpoint.pth")),
        # LRDropCheckpointSaver(("fold"+str(fold)+"_checkpoint_e{epoch}.pth")),
        ModelFreezer(),
        # EarlyStopper(10),
        TensorBoard(os.path.join('..', 'logs', config.folder, 'fold{}'.format(fold)))
    ]
    # if not num_channels_changed:
    #     callbacks.append(LastCheckpointSaver("fold"+str(fold)+"_checkpoint_rgb.pth", config.nb_epoch))

    hard_neg_miner = None#HardNegativeMiner(rate=10)
    # metrics = [('dr', dice_round)]

    trainer = PytorchTrain(estimator,
                           fold=fold,
                           callbacks=callbacks,
                           hard_negative_miner=hard_neg_miner)

    train_loader = PytorchDataLoader(TrainDataset(ds, train_idx, config, transforms=transforms),
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
    val_loader = PytorchDataLoader(ValDataset(val_ds, val_idx, config, transforms=val_transforms),
                                   batch_size=1,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=num_workers,
                                   pin_memory=True)

    trainer.fit(train_loader, val_loader, config.nb_epoch)
