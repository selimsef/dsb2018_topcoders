import random
import numpy as np

class Compose:
    def __init__(self, transforms, prob=1.0):
        self.transforms = [t for t in transforms if t is not None]
        self.prob = prob

    def __call__(self, **data):
        if random.random() < self.prob:

            for t in self.transforms:
                data = t(**data)
        return data


class OneOf:
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, **data):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            data = t(**data)
        return data


class OneOrOther:
    def __init__(self, first, second, prob=.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, **data):
        return self.first(**data) if random.random() < self.prob else self.second(**data)


class GrayscaleOrColor:
    def __init__(self, color_transform, grayscale_transform):
        self.color_transform = color_transform
        self.grayscale_transform = grayscale_transform

    def __call__(self, **data):
        image = data['image']
        grayscale = np.allclose(image[..., 0], image[..., 1], atol=0.001) and np.allclose(image[..., 1], image[..., 2], atol=0.001)
        if not grayscale:
            return self.color_transform(**data)
        else:
            return self.grayscale_transform(**data)