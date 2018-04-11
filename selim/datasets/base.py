
import os
import random
import time
from abc import abstractmethod

import cv2
import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import Iterator, load_img, img_to_array

from params import args


class BaseMaskDatasetIterator(Iterator):
    def __init__(self,
                 images_dir,
                 masks_dir,
                 labels_dir,
                 image_ids,
                 crop_shape,
                 preprocessing_function,
                 random_transformer=None,
                 batch_size=8,
                 shuffle=True,
                 image_name_template=None,
                 mask_template=None,
                 label_template=None,
                 padding=32,
                 seed=None,
                 grayscale_mask=False,
                 ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels_dir = labels_dir
        self.image_ids = image_ids
        self.image_name_template = image_name_template
        self.mask_template = mask_template
        self.label_template = label_template
        self.random_transformer = random_transformer
        self.crop_shape = crop_shape
        self.preprocessing_function = preprocessing_function
        self.padding = padding
        self.grayscale_mask = grayscale_mask
        if seed is None:
            seed = np.uint32(time.time() * 1000)

        super(BaseMaskDatasetIterator, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    @abstractmethod
    def transform_mask(self, mask, image):
        raise NotImplementedError

    def augment_and_crop_mask_image(self, mask, image, label, img_id, crop_shape):
        return mask, image, label

    def transform_batch_y(self, batch_y):
        return batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):
            id = self.image_ids[image_index]
            img_name = self.image_name_template.format(id=id)
            path = os.path.join(self.images_dir, img_name)
            image = np.array(img_to_array(load_img(path)), "uint8")
            mask_name = self.mask_template.format(id=id)
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            label = cv2.imread(os.path.join(self.labels_dir, self.label_template.format(id=id)), cv2.IMREAD_UNCHANGED)
            if args.use_full_masks:
                mask[...,0] = (label > 0) * 255
            if self.crop_shape is not None:
                crop_mask, crop_image, crop_label = self.augment_and_crop_mask_image(mask, image, label, id, self.crop_shape)
                data = self.random_transformer(image=np.array(crop_image, "uint8"), mask=np.array(crop_mask, "uint8"))
                crop_image, crop_mask = data['image'], data['mask']
                if len(np.shape(crop_mask)) == 2:
                    crop_mask = np.expand_dims(crop_mask, -1)
                crop_mask = self.transform_mask(crop_mask, crop_image)
                batch_x.append(crop_image)
                batch_y.append(crop_mask)
            else:
                x0, x1, y0, y1 = 0, 0, 0, 0
                if (image.shape[1] % 32) != 0:
                    x0 = int((32 - image.shape[1] % 32) / 2)
                    x1 = (32 - image.shape[1] % 32) - x0
                if (image.shape[0] % 32) != 0:
                    y0 = int((32 - image.shape[0] % 32) / 2)
                    y1 = (32 - image.shape[0] % 32) - y0
                image = np.pad(image, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                mask = np.pad(mask, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                batch_x.append(image)
                mask = self.transform_mask(mask, image)

                batch_y.append(mask)
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        if self.preprocessing_function:
            batch_x = imagenet_utils.preprocess_input(batch_x, mode=self.preprocessing_function)
        return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)

    def transform_batch_x(self, batch_x):
        return batch_x


    def next(self):

        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


