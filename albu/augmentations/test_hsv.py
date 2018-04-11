from transforms import *
from functional import add_channel
from composition import OneOf
from scipy.misc import imread
import os
from random import shuffle
import cv2

root = r'D:\tmp\bowl\train_imgs\images_all5'
masks = r'D:\tmp\bowl\train_imgs\masks_all6'
# augs = Compose([
#     CLAHE(clipLimit=5, prob=1),
#     InvertImg(prob=1),
#     Remap(prob=1),
#     RandomRotate90(prob=1),
#     Transpose(prob=1),
#     Blur(blur_limit=7, prob=1),
#     ElasticTransform(prob=1),
#     Distort1(prob=1),
#     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.30, rotate_limit=45, prob=1),
#     HueSaturationValue(prob=1),
#     ChannelShuffle(prob=1)
# ])
# augs = Distort2(prob=1)
# augs = aug_oneof()
# augs = GaussNoise(1.)
augs = aug_victor()
data = os.listdir(root)
shuffle(data)
for fn in data:
    im = imread(os.path.join(root, fn), mode='RGB')
    mask = imread(os.path.join(masks, fn), mode='RGB')
    cv2.imshow('before', im)
    dat = augs(image=im, mask=mask)
    print(dat['image'].shape)
    # cv2.imshow('after', cv2.cvtColor(dat['image'][...,:3], cv2.COLOR_RGB2BGR))
    cv2.imshow('after', dat['image'][...,:3])
    cv2.imshow('mask', dat['mask'])
    cv2.waitKey()
