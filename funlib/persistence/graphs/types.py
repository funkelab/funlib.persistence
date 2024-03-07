from dataclasses import dataclass


@dataclass
class Array:
    dtype: type | str
    size: int


def type_to_str(type):
    if isinstance(type, Array):
        return f"Array({type_to_str(type.dtype)}, {type.size})"
    else:
        return type.__name__
