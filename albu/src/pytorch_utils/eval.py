import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# torch.backends.cudnn.benchmark = True
import tqdm
from augmentations.tta import transforms as TTA
from augmentations.transforms import ToTensor


from dataset.neural_dataset import SequentialDataset
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader


class flip:
    FLIP_NONE=0
    FLIP_LR=1
    FLIP_FULL=2


def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(columns)))).cuda())
    return batch.index_select(3, index)


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(rows)))).cuda())
    return batch.index_select(2, index)


def to_numpy(batch):
    classes = batch.size(1)
    if isinstance(batch, tuple):
        batch = batch[0]
    # batch = F.softmax(batch, dim=1)
    data = np.moveaxis(batch.data.cpu().numpy(), 1, -1)
    return data


def predict(model, batch, flips=flip.FLIP_NONE):
    batch = torch.autograd.Variable(batch, volatile=True).cuda()
    pred1 = model(batch)
    if flips > flip.FLIP_NONE:
        pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        masks = [pred1, pred2]
        if flips > flip.FLIP_LR:
            pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
            pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])
        masks = [F.sigmoid(mask) for mask in masks]
        new_mask = torch.mean(torch.stack(masks, 0), 0)
        return to_numpy(new_mask)
    return to_numpy(F.sigmoid(pred1))

def predict8tta(model, batch, sigmoid):
    ret = []
    for cls in TTA:
        ret.append(cls(sigmoid)(model, batch))
    scale_tta = False
    if scale_tta:
        for scale in [0.8, 1.25]:
            data = np.moveaxis(np.squeeze(batch.numpy()[0]), 0, -1)
            srows, scols = data.shape[:2]
            data = cv2.resize(data, (0, 0), fx=scale, fy=scale)
            rows, cols = data.shape[:2]
            data = cv2.copyMakeBorder(data, 0, (32-rows%32), 0, (32-cols%32), cv2.BORDER_REFLECT)
            data = np.expand_dims(np.moveaxis(data, -1, 0), 0)
            data = torch.from_numpy(data)
            for cls in TTA:
                r = (cls(sigmoid)(model, data))
                r = np.moveaxis(np.squeeze(r), 0, -1)
                r = r[:rows, :cols, ...]
                r = cv2.resize(r, (scols, srows))
                r = np.expand_dims(np.moveaxis(r, -1, 0), 0)
                ret.append(r)
    return np.moveaxis(np.mean(ret, axis=0), 1, -1)


def read_model(project, fold):
    # model = nn.DataParallel(torch.load(os.path.join('..', 'weights', project, 'fold{}_best.pth'.format(fold))))
    model = nn.DataParallel(torch.load(os.path.join('..', 'weights', project, 'fold{}_best.pth'.format(fold))))
    model.eval()
    return model


class Evaluator:
    def __init__(self, config, ds, test=False, flips=0, num_workers=0, border=12, val_transforms=None):
        self.config = config
        self.ds = ds
        self.test = test
        self.flips = flips
        self.num_workers = num_workers
        self.image_needed = False

        self.current_prediction = None
        self.need_to_save = False
        self.border = border
        self.folder = config.folder

        self.save_dir = os.path.join(self.config.results_dir + ('_test' if self.test else ''), self.folder)
        self.val_transforms = val_transforms
        os.makedirs(self.save_dir, exist_ok=True)

    def predict(self, fold, val_indexes):
        prefix = ('fold' + str(fold) + "_") if self.test else ""
        val_dataset = SequentialDataset(self.ds, val_indexes, stage='test', config=self.config, transforms=self.val_transforms)
        val_dl = PytorchDataLoader(val_dataset, batch_size=self.config.predict_batch_size, num_workers=self.num_workers, drop_last=False)
        model = read_model(self.folder, fold)
        pbar = tqdm.tqdm(val_dl, total=len(val_dl))
        for data in pbar:
            samples = data['image']
            # predicted = predict(model, samples, flips=self.flips)
            predicted = predict8tta(model, samples, self.config.sigmoid)
            self.process_batch(predicted, model, data, prefix=prefix)
        self.post_predict_action(prefix=prefix)

    def cut_border(self, image):
        if image is None:
            return None
        return image if not self.border else image[self.border:-self.border, self.border:-self.border, ...]

    def on_image_constructed(self, name, prediction, prefix=""):
        prediction = self.cut_border(prediction)
        prediction = np.squeeze(prediction)
        self.save(name, prediction, prefix=prefix)

    def save(self, name, prediction, prefix=""):
        raise NotImplementedError

    def process_batch(self, predicted, model, data, prefix=""):
        raise NotImplementedError

    def post_predict_action(self, prefix):
        pass
