import os

from .abstract_image_provider import AbstractImageProvider
import numpy as np


class ReadingImageProvider(AbstractImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, image_suffix=None, has_alpha=False):
        super(ReadingImageProvider, self).__init__(image_type, fn_mapping, has_alpha=has_alpha)
        self.im_names = os.listdir(paths['images'])
        if image_suffix is not None:
            self.im_names = [n for n in self.im_names if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        indexes = {os.path.splitext(name)[0]: idx for idx, name in enumerate(self.im_names)}
        ret = [indexes[name] for name in names if name in indexes]
        return ret

    def __getitem__(self, item):
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.has_alpha)

    def __len__(self):
        return len(self.im_names)


class CachingImageProvider(ReadingImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, image_suffix=None, has_alpha=False):
        super().__init__(image_type, paths, fn_mapping, image_suffix, has_alpha=has_alpha)
        self.cache = {}

    def __getitem__(self, item):
        if item not in self.cache:
            data = super().__getitem__(item)
            self.cache[item] = data
        return self.cache[item]

class InFolderImageProvider(ReadingImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, image_suffix=None, has_alpha=False):
        super().__init__(image_type, paths, fn_mapping, image_suffix, has_alpha)

    def __getitem__(self, item):
        return self.image_type(self.paths, os.path.join(self.im_names[item], 'images', self.im_names[item] + '.png'), self.fn_mapping, self.has_alpha)
