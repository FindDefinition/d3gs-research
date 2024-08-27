# from tensorpc

from typing import Any
from pydantic_core import core_schema
from pydantic import (
    GetCoreSchemaHandler, )

class UniqueTreeId:
    # format: length1_length2_length3-part1-part2-part3
    # reversed format: part1-part2-part3-length1_length2_length3
    # part names may contains splitter '::', so we need lengths to split
    # splitter is only used for better readability, it's not necessary.
    def __init__(self, uid: str, splitter_length: int = 1, reversed: bool = True) -> None:
        self.uid_encoded = uid
        self._reversed = reversed
        # init_parts = uid.split("|")
        if reversed:
            splitter_first_index = uid.rfind("-")
        else:
            splitter_first_index = uid.find("-")
        if splitter_first_index == -1:
            # empty uid, means uid must be ""
            assert len(
                uid
            ) == 0, f"uid should be empty if no splitter exists, but got {uid}"
            self.parts: list[str] = []
            uid_part = ""
            lengths: list[int] = []
        else:
            if reversed:
                uid_part = uid[:splitter_first_index]
                length_part = uid[splitter_first_index + 1:]
            else:
                length_part = uid[:splitter_first_index]
                uid_part = uid[splitter_first_index + 1:]
            lengths = [int(n) for n in length_part.split("_")]
            assert sum(lengths) == len(uid_part) - splitter_length * (
                len(lengths) - 1), f"{uid} not valid, {lengths}, {uid_part}"
        start = 0
        self.parts: list[str] = []
        for l in lengths:
            self.parts.append(uid_part[start:start + l])
            start += l + splitter_length
        self.splitter_length = splitter_length

    def empty(self):
        return len(self.uid_encoded) == 0

    @classmethod
    def from_parts(cls,
                   parts: list[str],
                   splitter: str = "-",
                   reversed: bool = True) -> "UniqueTreeId":
        if len(parts) == 0:
            return cls("", len(splitter))
        if reversed:
            return cls(
                "-".join(parts) + "-" + "_".join([str(len(p))
                                                 for p in parts]),
                len(splitter), reversed)
        else:
            return cls(
                "_".join([str(len(p))
                        for p in parts]) + "-" + splitter.join(parts),
                len(splitter))

    def __repr__(self) -> str:
        return f"UniqueTreeId({self.uid_encoded})"

    def __hash__(self) -> int:
        return hash(self.uid_encoded)

    def append_part(self, part: str, splitter: str = "-") -> "UniqueTreeId":
        return UniqueTreeId.from_parts(self.parts + [part], splitter, self._reversed)

    def pop(self):
        return UniqueTreeId.from_parts(self.parts[:-1], "-" * self.splitter_length, self._reversed)

    def reverse(self) -> "UniqueTreeId":
        return UniqueTreeId.from_parts(self.parts, "-" * self.splitter_length,
                                       not self._reversed)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, UniqueTreeId):
            return False
        if self._reversed ^ o._reversed:
            return self.parts == o.parts
        return self.uid_encoded == o.uid_encoded

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __add__(self, other: "UniqueTreeId | str") -> "UniqueTreeId":
        if isinstance(other, str):
            return UniqueTreeId.from_parts(self.parts + [other], "-", self._reversed)
        return UniqueTreeId.from_parts(self.parts + other.parts, "-", self._reversed)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, str):
            return UniqueTreeId(v)
        if not isinstance(v, UniqueTreeId):
            raise ValueError('undefined required, but get', type(v))
        return v

    def startswith(self, other: "UniqueTreeId") -> bool:
        if len(self.parts) < len(other.parts):
            return False
        for i in range(len(other.parts)):
            if self.parts[i] != other.parts[i]:
                return False
        return True

    def common_prefix(self, other: "UniqueTreeId") -> "UniqueTreeId":
        i = 0
        while i < len(self.parts) and i < len(
                other.parts) and self.parts[i] == other.parts[i]:
            i += 1
        return UniqueTreeId.from_parts(self.parts[:i])

    def common_prefix_index(self, other: "UniqueTreeId") -> int:
        i = 0
        while i < len(self.parts) and i < len(
                other.parts) and self.parts[i] == other.parts[i]:
            i += 1
        return i

