
import inspect
from typing import Any, Callable, Dict, Hashable, List, Optional, Generic, Type, TypeVar, Union

T = TypeVar("T", bound=Union[Type, Callable])


class HashableRegistry(Generic[T]):

    def __init__(self, allow_duplicate: bool = True):
        self.global_dict: Dict[Hashable, T] = {}
        self.allow_duplicate = allow_duplicate

    def register_no_key(self, func: T):

        def wrapper(func: T) -> T:
            key_ = func.__name__
            if not self.allow_duplicate and key_ in self.global_dict:
                raise KeyError("key {} already exists".format(key_))
            self.global_dict[key_] = func
            return func

        return wrapper(func)

    def register_with_key(self, key: str):
        def wrapper(func: T) -> T:
            key_ = key
            if not self.allow_duplicate and key_ in self.global_dict:
                raise KeyError("key {} already exists".format(key_))
            self.global_dict[key_] = func
            return func

        return wrapper

    def __contains__(self, key: Hashable):
        return key in self.global_dict

    def __getitem__(self, key: Hashable):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()

