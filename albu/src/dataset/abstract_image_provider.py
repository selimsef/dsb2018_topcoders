from .abstract_image_type import AbstractImageType
from typing import Type, Dict, AnyStr, Callable

class AbstractImageProvider:
    def __init__(self, image_type: Type[AbstractImageType], fn_mapping: Dict[AnyStr, Callable], has_alpha=False):
        self.image_type = image_type
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


