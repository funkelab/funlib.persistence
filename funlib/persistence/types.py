from dataclasses import dataclass


@dataclass
class Vec:
    dtype: type | str
    size: int


def type_to_str(type):
    if isinstance(type, Vec):
        return f"Vec({type_to_str(type.dtype)}, {type.size})"
    else:
        return type.__name__
