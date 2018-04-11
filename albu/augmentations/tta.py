import numpy as np
import cv2
import os
import torch
from torch.nn import functional as F


class TTAOp:
    def __init__(self, sigmoid=True):
        self.sigmoid = sigmoid

    def __call__(self, model, batch):
        forwarded = torch.autograd.Variable(torch.from_numpy(self.forward(batch.numpy())), volatile=True).cuda()
        return self.backward(self.to_numpy(model(forwarded)))

    def forward(self, img):
        raise NotImplementedError

    def backward(self, img):
        raise NotImplementedError

    def to_numpy(self, batch):
        if self.sigmoid:
            batch = F.sigmoid(batch)
        else:
            batch = F.softmax(batch, dim=1)
        data = batch.data.cpu().numpy()
        return data

class BasicTTAOp(TTAOp):
    @staticmethod
    def op(img):
        raise NotImplementedError

    def forward(self, img):
        return self.op(img)

    def backward(self, img):
        return self.forward(img)

class Nothing(BasicTTAOp):
    @staticmethod
    def op(img):
        return img

class HFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=2))


class VFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=3))


class Transpose(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(img.transpose(0, 1, 3, 2))


def chain_op(data, operations):
    for op in operations:
        data = op.op(data)
    return data


class ChainedTTA(TTAOp):
    @property
    def operations(self):
        raise NotImplementedError

    def forward(self, img):
        return chain_op(img, self.operations)

    def backward(self, img):
        return chain_op(img, reversed(self.operations))

class HVFlip(ChainedTTA):
    @property
    def operations(self):
        return [HFlip, VFlip]


class TransposeHFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip]


class TransposeVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, VFlip]


class TransposeHVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip, VFlip]

transforms = [Nothing, HFlip, VFlip, Transpose, HVFlip, TransposeHFlip, TransposeVFlip, TransposeHVFlip]

if __name__ == "__main__":
    root = r'D:\tmp\bowl\train_imgs\images'
    imgs = os.listdir(root)[:2]
    imgs = [cv2.imread(os.path.join(root, im)) / 255. for im in imgs]
    data = torch.from_numpy(np.moveaxis(np.stack((imgs)).astype(np.float32), -1, 1))
    for cls in transforms:
        flip = cls()
        ret = flip(lambda x: x, data)
        assert np.allclose(ret, data)
