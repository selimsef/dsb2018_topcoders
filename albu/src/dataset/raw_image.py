import os
from scipy.misc import imread
import cv2

from dataset.abstract_image_type import AbstractImageType, AlphaNotAvailableException


class RawImageType(AbstractImageType):
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        self.im = imread(os.path.join(self.paths['images'], self.fn), mode='RGB')
        if '646f5e00a2db3add97fb80a83ef3c07edd1b17b1b0d47c2bd650cdcab9f322c0' in fn:
            self.im = cv2.imread(os.path.join(self.paths['images'], self.fn), cv2.IMREAD_COLOR)
        # self.im = 255 - self.im
        # self.clahe = CLAHE(1)
        # self.im = self.clahe(image=self.im)['image']

    def read_image(self):
        im = self.im[...,:-1] if self.has_alpha else self.im
        return self.finalyze(im)

    def read_mask(self):
        path = os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn))
        mask = imread(path, mode='L')
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[...,-1])

    def read_label(self):
        path = os.path.join(self.paths['labels'], self.fn_mapping['labels'](self.fn))
        label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return self.finalyze(label)

    def finalyze(self, data):
        return data


