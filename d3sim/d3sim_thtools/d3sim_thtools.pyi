from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue, enum
import torch
class PyTorchTools:
    @staticmethod
    def torch2tensor(ten: torch.Tensor): 
        """
        Args:
            ten: 
        """
        ...
    @staticmethod
    def tensor2torch(ten, clone: bool = True, cast_uint_to_int: bool = False) -> torch.Tensor: 
        """
        Args:
            ten: 
            clone: 
            cast_uint_to_int: 
        """
        ...
    @staticmethod
    def mps_get_default_command_buffer() -> int: ...
    @staticmethod
    def mps_get_default_dispatch_queue() -> int: ...
