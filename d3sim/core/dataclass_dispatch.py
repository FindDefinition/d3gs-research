from pydantic.dataclasses import dataclass
from pydantic import Field as field

from dataclasses import asdict, is_dataclass, fields, replace

class PyDanticConfigForAnyObject:
    arbitrary_types_allowed = True
