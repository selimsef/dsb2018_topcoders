import random
import numpy as np
from augmentations.composition import Compose, OneOf
import augmentations.functional as F
from imgaug import augmenters as iaa

def to_tuple(param, low=None):
    if isinstance(param, tuple):
        return param
    else:
        return (-param if low is None else low, param)


class BasicTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **kwargs):
        if random.random() < self.prob:
            params = self.get_params()
            return {k: self.apply(a, **params) if k in self.targets else a for k, a in kwargs.items()}
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

class BasicIAATransform(BasicTransform):
    def __init__(self, prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Noop()
        self.deterministic_processor = iaa.Noop()

    def __call__(self, **kwargs):
        self.deterministic_processor = self.processor.to_deterministic()
        return super().__call__(**kwargs)

    def apply(self, img, **params):
        return self.deterministic_processor.augment_image(img)


class DualTransform(BasicTransform):
    """
    transfrom for segmentation task
    """
    @property
    def targets(self):
        return 'image', 'mask'

class DualIAATransform(DualTransform, BasicIAATransform):
    pass


class ImageOnlyTransform(BasicTransform):
    """
    transforms applied to image only
    """
    @property
    def targets(self):
        return 'image'

class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass

class VerticalFlip(DualTransform):
    def apply(self, img, **params):
        return F.vflip(img)


class HorizontalFlip(DualTransform):
    def apply(self, img, **params):
        return F.hflip(img)


class Flip(DualTransform):
    def apply(self, img, d=0):
        return F.random_flip(img, d)

    def get_params(self):
        return {'d': random.randint(-1, 1)}


class Transpose(DualTransform):
    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    def apply(self, img, factor=0):
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        return {'factor': random.randint(0, 4)}


class Rotate(DualTransform):
    def __init__(self, limit=90, prob=.5):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, angle=0):
        return F.rotate(img, angle)

    def get_params(self):
        return {'angle': random.uniform(self.limit[0], self.limit[1])}


class ShiftScaleRotate(DualTransform):
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        super().__init__(prob)
        self.shift_limit = to_tuple(shift_limit)
        self.scale_limit = to_tuple(scale_limit)
        self.rotate_limit = to_tuple(rotate_limit)

    def apply(self, img, angle=0, scale=0, dx=0, dy=0):
        return F.shift_scale_rotate(img, angle, scale, dx, dy)

    def get_params(self):
        return {'angle': random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
                'scale': random.uniform(1+self.scale_limit[0], 1+self.scale_limit[1]),
                'dx': round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
                'dy': round(random.uniform(self.shift_limit[0], self.shift_limit[1]))}


class CenterCrop(DualTransform):
    def __init__(self, height, width, prob=0.5):
        super().__init__(prob)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)


class Distort1(DualTransform):
    def __init__(self, distort_limit=0.05, shift_limit=0.05, prob=0.5):
        super().__init__(prob)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.shift_limit = to_tuple(shift_limit)

    def apply(self, img, k=0, dx=0, dy=0):
        return F.distort1(img, k, dx, dy)

    def get_params(self):
        return {'k': random.uniform(self.distort_limit[0], self.distort_limit[1]),
                'dx': round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
                'dy': round(random.uniform(self.shift_limit[0], self.shift_limit[1]))}


class Distort2(DualTransform):
    def __init__(self, num_steps=5, distort_limit=0.3, prob=0.5):
        super().__init__(prob)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.prob = prob

    def apply(self, img, stepsx=[], stepsy=[]):
        return F.distort2(img, self.num_steps, stepsx, stepsy)

    def get_params(self):
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        return {
            'stepsx': stepsx,
            'stepsy': stepsy
        }


class ElasticTransform(DualTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, prob=0.5):
        super().__init__(prob)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma

    def apply(self, img, random_state=None):
        return F.elastic_transform_fast(img, self.alpha, self.sigma, self.alpha_affine, np.random.RandomState(random_state))

    def get_params(self):
        return {'random_state': np.random.randint(0, 10000)}


class HueSaturationValue(ImageOnlyTransform):
    def __init__(self, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, prob=0.5):
        # def __init__(self, hue_shift_limit=100, sat_shift_limit=50, val_shift_limit=10, prob=0.5, targets=('image')):
        super().__init__(prob)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0):
        assert image.dtype == np.uint8 or self.hue_shift_limit < 1.
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return{'hue_shift': np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
               'sat_shift': np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
               'val_shift': np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])}


class RGBShift(ImageOnlyTransform):
    def __init__(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, prob=0.5):
        super().__init__(prob)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0):
        return F.shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return{'r_shift': np.random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
               'g_shift': np.random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
               'b_shift': np.random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])}


class RandomBrightness(ImageOnlyTransform):
    def __init__(self, limit=0.2, prob=0.5):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2):
        return F.random_brightness(img, alpha)

    def get_params(self):
        return {"alpha": 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class RandomContrast(ImageOnlyTransform):
    def __init__(self, limit=.2, prob=.5):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2):
        return F.random_contrast(img, alpha)

    def get_params(self):
        return {"alpha": 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class Blur(ImageOnlyTransform):
    def __init__(self, blur_limit=7, prob=.5):
        super().__init__(prob)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            'ksize': np.random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        }

class MotionBlur(Blur):
    def apply(self, img, ksize=9):
        return F.motion_blur(img, ksize=ksize)


class MedianBlur(Blur):
    def apply(self, image, ksize=3):
        return F.median_blur(image, ksize)


class Remap(ImageOnlyTransform):
    backgrounds = {
        'gray': [216, 222, 222],
        'dark-gray': [206, 202, 202],
        'purple': [181, 159, 210],
        # 'pink': [211, 151, 204],
        'pink-gray': [235, 211, 235],
        # 'pink2': [208, 145, 202]
    }

    nuclei_max = {
        'gray': [80, 20, 105],
        'pink': [117, 12, 127],
        'purple': [82, 23, 192],
    }

    nuclei_center = {
        'gray': [149, 81, 168],
        'purple': [136, 107, 204],
        'pink': [170, 62, 176]
    }

    def apply(self, img, bg=[], center=[], max=[]):
        return F.remap_color(img, bg, center, max)

    def get_params(self):
        bg = random.choice(list(Remap.backgrounds.values()))
        center = random.choice(list(Remap.nuclei_center.values()))
        max = random.choice(list(Remap.nuclei_max.values()))
        return {
            'bg': bg,
            'center': center,
            'max': max
        }

class GaussNoise(ImageOnlyTransform):
    def __init__(self, var_limit=(10, 50), prob=.5):
        super().__init__(prob)
        self.var_limit = to_tuple(var_limit)

    def apply(self, img, var=30):
        return F.gauss_noise(img, var=var)

    def get_params(self):
        return {
            'var': np.random.randint(self.var_limit[0], self.var_limit[1])
        }


class CLAHE(ImageOnlyTransform):
    def __init__(self, clipLimit=4.0, tileGridSize=(8, 8), prob=0.5):
        super().__init__(prob)
        self.clipLimit = to_tuple(clipLimit, 1)
        self.tileGridSize = tileGridSize

    def apply(self, img, clipLimit=2):
        return F.clahe(img, clipLimit, self.tileGridSize)

    def get_params(self):
        return {"clipLimit": np.random.uniform(self.clipLimit[0], self.clipLimit[1])}


class IAAEmboss(ImageOnlyIAATransform):
    def __init__(self, alpha=(0.2, 0.5), strength=(0.2, 0.7), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Emboss(alpha, strength)


class IAASuperpixels(ImageOnlyIAATransform):
    '''
    may be slow
    '''
    def __init__(self, p_replace=0.1, n_segments=100, prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Superpixels(p_replace=p_replace, n_segments=n_segments)


class IAASharpen(ImageOnlyIAATransform):
    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Sharpen(alpha, lightness)


class IAAAdditiveGaussianNoise(ImageOnlyIAATransform):
    def __init__(self, loc=0, scale=(0.01*255, 0.05*255), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.AdditiveGaussianNoise(loc, scale)


class IAAPiecewiseAffine(DualIAATransform):
    def __init__(self, scale=(0.03, 0.05), nb_rows=4, nb_cols=4, prob=.5):
        super().__init__(prob)
        self.processor = iaa.PiecewiseAffine(scale, nb_rows, nb_cols)

class IAAPerspective(DualIAATransform):
    def __init__(self, scale=(0.05, 0.1), prob=.5):
        super().__init__(prob)
        self.processor = iaa.PerspectiveTransform(scale)


class ChannelShuffle(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.channel_shuffle(img)


class InvertImg(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.invert(img)


class ToThreeChannelGray(ImageOnlyTransform):
    def __init__(self, prob=1.):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.to_three_channel_gray(img)


class ToGray(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.to_gray(img)


class RandomLine(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.random_polosa(img)


class AddChannel(ImageOnlyTransform):
    def __init__(self, prob=1.):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.add_channel(img)


class FixMasks(BasicTransform):
    def __init__(self, prob=1.):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.fix_mask(img)

    @property
    def targets(self):
        return ('mask',)


class ToTensor(BasicTransform):
    def __init__(self, num_classes=1, sigmoid=True):
        super().__init__(1.)
        self.num_classes = num_classes
        self.sigmoid = sigmoid

    def __call__(self, **kwargs):
        kwargs.update({'image': F.img_to_tensor(kwargs['image'])})
        if 'mask' in kwargs.keys():
            kwargs.update({'mask': F.mask_to_tensor(kwargs['mask'], self.num_classes, sigmoid=self.sigmoid)})
        return kwargs

    @property
    def targets(self):
        raise NotImplementedError

def aug_hardcore(prob=.95):
    return Compose([
        CLAHE(clipLimit=2, prob=0.35),
        GaussNoise(),
        ToGray(prob=0.25),
        InvertImg(prob=0.2),
        Remap(prob=0.4),
        RandomRotate90(),
        Flip(),
        Transpose(),
        Blur(blur_limit=3, prob=.4),
        RandomContrast(prob=.2),
        RandomBrightness(prob=.2),
        ElasticTransform(prob=0.3),
        Distort1(prob=0.3),
        Distort2(prob=.1),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=45, prob=.7),
        HueSaturationValue(),
        ChannelShuffle(prob=.2)
    ])

def aug_mega_hardcore(scale_limits=(-0.45, 0.45)):
    return Compose([
        OneOf([
            CLAHE(clipLimit=2, prob=.5),
            IAASharpen(prob=.25),
            IAAEmboss(prob=.25)
        ], prob=.35),
        OneOf([
            IAAAdditiveGaussianNoise(prob=.3),
            GaussNoise(prob=.7),
            ], prob=.5),    
        ToGray(prob=.25),
        InvertImg(prob=.2),
        Remap(prob=.4),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            MotionBlur(prob=.2),
            MedianBlur(blur_limit=3, prob=.3),
            Blur(blur_limit=3, prob=.5),
        ], prob=.4),
        OneOf([
            RandomContrast(prob=.5),
            RandomBrightness(prob=.5),
        ], prob=.4),
        ShiftScaleRotate(shift_limit=.0625, scale_limit=scale_limits, rotate_limit=45, prob=.7),
        OneOf([
            Distort1(prob=.2),
            Distort2(prob=.2),
            ElasticTransform(prob=.2),
            IAAPerspective(prob=.2),
            IAAPiecewiseAffine(prob=.2),
        ], prob=.6),
        HueSaturationValue(prob=.5),
        ChannelShuffle(prob=.2)
    ], prob=.97)