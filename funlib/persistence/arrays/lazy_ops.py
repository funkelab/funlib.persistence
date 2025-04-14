from typing import Callable, Union

from funlib.geometry import Roi

LazyOp = Union[slice, Callable, Roi]
