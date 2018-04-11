import random
import numpy as np


class Compose:
    def __init__(self, transforms, prob=1.):
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
        transforms_probs = [t.prob for t in transforms]
        s = sum(transforms_probs)
        self.transforms_probs = [t / s for t in transforms_probs]

    def __call__(self, **data):
        if random.random() < self.prob:
            t = np.random.choice(self.transforms, p=self.transforms_probs)
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
