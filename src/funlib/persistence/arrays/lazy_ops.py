from typing import Callable, Union

import numpy as np
from funlib.geometry import Roi

LazyOp = Union[slice, int, tuple[int | slice | list[int] | np.ndarray, ...], Callable, Roi]
