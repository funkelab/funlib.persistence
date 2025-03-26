from functools import reduce

import numpy as np
import pytest
from dask.array.optimization import fuse_slice


def test_slice_chaining():
    def combine_slices(*slices):
        return reduce(fuse_slice, slices)

    base = np.s_[::2, :, :4]

    # chain with index expressions

    s1 = combine_slices(base, np.s_[0])
    assert s1 == np.s_[0, :, :4]

    s2 = combine_slices(s1, np.s_[1])
    assert s2 == np.s_[0, 1, :4]

    # chain with index arrays

    s1 = combine_slices(base, np.s_[[0, 1, 1, 2, 3, 5], :])
    assert s1 == np.s_[[0, 2, 2, 4, 6, 10], 0:, :4]

    # ...and another index array
    with pytest.raises(NotImplementedError):
        # this is not supported because the combined indexing
        # operation would not behave the same as the individual
        # indexing operations performed in sequence
        combine_slices(s1, np.s_[[0, 3], 2])

    # ...and a slice() expression
    s22 = combine_slices(s1, np.s_[1:4])
    assert s22 == np.s_[[2, 2, 4], 0:, :4]

    # chain with slice expressions

    s1 = combine_slices(base, np.s_[10:20, ::2, 0])
    assert s1 == np.s_[20:40:2, 0::2, 0]
