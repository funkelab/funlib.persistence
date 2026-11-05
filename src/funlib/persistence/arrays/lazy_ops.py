from collections.abc import Callable

import numpy as np
from funlib.geometry import Roi

LazyOp = slice | int | tuple[int | slice | list[int] | np.ndarray, ...] | Callable | Roi
