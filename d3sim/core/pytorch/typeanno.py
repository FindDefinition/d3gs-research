from typing_extensions import TypeVarTuple, Unpack
from typing import Any, Generic, Literal, TypeAlias

Ts = TypeVarTuple('Ts')

Shape: TypeAlias = Literal[Unpack[Ts]]