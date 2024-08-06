
from typing import Any, Sequence
from d3sim.core import dataclass_dispatch as dataclasses
from pydantic import field_validator, model_validator
from typing_extensions import Self 
import torch
import numpy as np 
from typing_extensions import Literal, Annotated, NotRequired, get_origin, get_args, get_type_hints

from d3sim.core.arrcheck.tensor import ArrayCheckBase, ArrayValidator, check_shape_of_meta_shape

def lenient_issubclass(cls: Any,
                       class_or_tuple: Any) -> bool:  # pragma: no cover
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)

def is_annotated(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Annotated)

def extract_annotated_type_and_meta(ann_type: Any) -> tuple[Any, Any]:
    if is_annotated(ann_type):
        annometa = ann_type.__metadata__
        ann_type = get_args(ann_type)[0]
        return ann_type, annometa
    return ann_type, None


@dataclasses.dataclass
class DataClassWithArrayCheck:
    @model_validator(mode='after')
    def _validator_post(self) -> Self:
        annos = get_type_hints(type(self), include_extras=True)
        all_metas: list[ArrayCheckBase] = []
        all_real_shapes: list[Sequence[int]] = []
        for field in dataclasses.fields(self):
            if field.name in annos:
                anno = annos[field.name]
                _, annometa = extract_annotated_type_and_meta(anno)
                if annometa is not None:
                    for meta in annometa:
                        if isinstance(meta, ArrayValidator):
                            all_metas.append(meta.meta) 
                            all_real_shapes.append(getattr(self, field.name).shape)
        check_shape_of_meta_shape([meta.shape for meta in all_metas], all_real_shapes)
        return self


def __main():
    from d3sim.core.arrcheck import ArrayCheck, Float32, Float64
    @dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
    class Dev(DataClassWithArrayCheck):
        a: Annotated[torch.Tensor, ArrayCheck([-1, "N"], Float32 | Float64)]
        b: Annotated[torch.Tensor, ArrayCheck([-1, "N"], Float32 | Float64)]

    a = Dev(a=torch.zeros(3, 3), b=torch.zeros(3, 3))
    a_err = Dev(a=torch.zeros(3, 3), b=torch.zeros(3, 3))


if __name__ == "__main__":
    __main()