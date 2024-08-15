from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue, enum
class DeviceSort:
    @staticmethod
    def get_sort_workspace_size(keys_in, keys_out, values_in, values_out, bit_start: int, bit_end: int, double_buffer: bool) -> int: 
        """
        Args:
            keys_in: 
            keys_out: 
            values_in: 
            values_out: 
            bit_start: 
            bit_end: 
            double_buffer: 
        """
        ...
    @staticmethod
    def do_radix_sort_with_bit_range(keys_in, keys_out, values_in, values_out, workspace, bit_start: int, bit_end: int, double_buffer: bool, stream_int: int) -> Tuple[bool, bool]: 
        """
        Args:
            keys_in: 
            keys_out: 
            values_in: 
            values_out: 
            workspace: 
            bit_start: 
            bit_end: 
            double_buffer: 
            stream_int: 
        """
        ...
    @staticmethod
    def get_higher_msb(n: int) -> int: 
        """
        Args:
            n: 
        """
        ...
