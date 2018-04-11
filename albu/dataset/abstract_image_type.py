import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np


class AlphaNotAvailableException(Exception):
    pass

class AbstractImageType:
    def __init__(self, paths, fn, fn_mapping, has_alpha=False):
        self.paths = paths
        self.fn = fn
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping
        self.cache = {}

    @property
    def image(self):
        if 'image' not in self.cache:
            self.cache['image'] = self.read_image()
        return np.copy(self.cache['image'])

    @property
    def mask(self):
        if 'mask' not in self.cache:
            self.cache['mask'] = self.read_mask()
        return np.copy(self.cache['mask'])

    @property
    def alpha(self):
        if not self.has_alpha:
            raise AlphaNotAvailableException
        if 'alpha' not in self.cache:
            self.cache['alpha'] = self.read_alpha()
        return self.cache['alpha']

    @property
    def label(self):
        if 'label' not in self.cache:
            self.cache['label'] = self.read_label()
        return np.copy(self.cache['label'])


    def read_alpha(self):
        raise NotImplementedError

    def read_image(self):
        raise NotImplementedError

    def read_mask(self):
        raise NotImplementedError

    def read_label(self):
        raise NotImplementedError

    def reflect_border(self, image, b=12):
        return cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_REFLECT)

    def pad_image(self, image, rows, cols):
        channels = image.shape[2] if len(image.shape) > 2 else None
        if image.shape[:2] != (rows, cols):
            empty_x = np.zeros((rows, cols, channels), dtype=image.dtype) if channels else np.zeros((rows, cols), dtype=image.dtype)
            empty_x[0:image.shape[0],0:image.shape[1],...] = image
            image = empty_x
        return image

    def finalyze(self, image):
        return self.reflect_border(image)


