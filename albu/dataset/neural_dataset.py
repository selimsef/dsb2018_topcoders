import random

import numpy as np

from augmentations.composition import Compose
from augmentations.transforms import ToTensor
from dataset.abstract_image_provider import AbstractImageProvider
from .image_cropper import ImageCropper, DVCropper


class Dataset:
    def __init__(self, image_provider: AbstractImageProvider, image_indexes, config, stage='train', transforms=None):
        self.pad = 0 if stage=='train' else config.test_pad
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(image_indexes, list) else image_indexes.tolist()
        if stage != 'train' and len(self.image_indexes) % 2: #todo bugreport it
            self.image_indexes += [self.image_indexes[-1]]
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.config = config
        normalize = {'mean': [124 / 255, 117 / 255, 104 / 255],
                     'std': [1 / (.0167 * 255)] * 3}
        self.transforms = Compose([transforms, ToTensor(config.num_classes, config.sigmoid, normalize)])
        self.croppers = {}

    def __getitem__(self, item):
        raise NotImplementedError

    def get_cropper(self, image_id, val=False):
        #todo maybe cache croppers for different sizes too speedup if it's slow part?
        if image_id not in self.croppers:
            image = self.image_provider[image_id].image
            rows, cols = image.shape[:2]
            if self.config.ignore_target_size and val:
                assert self.config.predict_batch_size == 1
                target_rows, target_cols = rows, cols
            else:
                target_rows, target_cols = self.config.target_rows, self.config.target_cols
            cropper = ImageCropper(rows, cols,
                                   target_rows, target_cols,
                                   self.pad)
            self.croppers[image_id] = cropper
        return self.croppers[image_id]


class TrainDataset(Dataset):
    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=None, partly_sequential=False):
        super(TrainDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')
        self.partly_sequential = partly_sequential
        self.inner_idx = 9
        self.idx = 0
        masks = []
        labels = []
        # for im_idx in self.image_indexes:
        #     item = self.image_provider[im_idx]
        #     masks.append(item.mask)
        #     labels.append(item.label)
        # self.dv_cropper = DVCropper(masks, labels, config.target_rows, config.target_cols)


    def __getitem__(self, idx):
        if self.partly_sequential:
            #todo rewrite somehow better
            if self.inner_idx > 8:
                self.idx = idx
                self.inner_idx = 0
            self.inner_idx += 1
            im_idx = self.image_indexes[self.idx % len(self.image_indexes)]
        else:
            im_idx = self.image_indexes[idx % len(self.image_indexes)]

        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        sx, sy = cropper.random_crop_coords()
        if cropper.use_crop and self.image_provider.has_alpha:
            for i in range(10):
                alpha = cropper.crop_image(item.alpha, sx, sy)
                if np.mean(alpha) > 5:
                    break
                sx, sy = cropper.random_crop_coords()
            else:
                return self.__getitem__(random.randint(0, len(self.image_indexes)))

        im = cropper.crop_image(item.image, sx, sy)
        mask = cropper.crop_image(item.mask, sx, sy)
        # im, mask, lbl = item.image, item.mask, item.label
        # im, mask = self.dv_cropper.strange_method(idx % len(self.image_indexes), im, mask, lbl, sx, sy)
        data = {'image': im, 'mask': mask, 'image_name': item.fn}
        return self.transforms(**data)

    def __len__(self):
        return len(self.image_indexes) * max(self.config.epoch_size, 1) # epoch size is len images

class SequentialDataset(Dataset):
    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=None):
        super(SequentialDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys.update({'geometry'})

    def init_good_tiles(self):
        self.good_tiles = []
        for im_idx in self.image_indexes:
            cropper = self.get_cropper(im_idx, val=True)
            positions = cropper.positions
            if self.image_provider.has_alpha:
                item = self.image_provider[im_idx]
                alpha_generator = cropper.sequential_crops(item.alpha)
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))

    def prepare_image(self, item, cropper, sx, sy):
        im = cropper.crop_image(item.image, sx, sy)
        rows, cols = item.image.shape[:2]
        geometry = {'rows': rows, 'cols': cols, 'sx': sx, 'sy': sy}
        data = {'image': im, 'image_name': item.fn, 'geometry': geometry}
        return data

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)

        return self.transforms(**data)

    def __len__(self):
        return len(self.good_tiles)


class ValDataset(SequentialDataset):
    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=None):
        super(ValDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx):
        im_idx, sx, sy = self.good_tiles[idx]
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)

        mask = cropper.crop_image(item.mask, sx, sy)
        data.update({'mask': mask})
        return self.transforms(**data)
