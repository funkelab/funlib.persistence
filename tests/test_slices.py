import numpy as np
from funlib.persistence.arrays.slices import chain_slices


def test_slice_chaining():

    base = np.s_[::2, 0, :4]

    # chain with index expressions

    s1 = chain_slices(base, np.s_[0])
    assert s1 == np.s_[0, 0, :4]

    s2 = chain_slices(s1, np.s_[1])
    assert s2 == np.s_[0, 0, 1]

    # chain with index arrays

    s1 = chain_slices(base, np.s_[[0, 1, 1, 2, 3, 5], :])
    assert s1 == np.s_[[0, 2, 2, 4, 6, 10], 0, :4]

    # ...and another index array
    s21 = chain_slices(s1, np.s_[[0, 3], :])
    assert s21 == np.s_[[0, 4], 0, :4]

    # ...and a slice() expression
    s22 = chain_slices(s1, np.s_[1:4])
    assert s22 == np.s_[[2, 2, 4], 0, :4]

    # chain with slice expressions

    s1 = chain_slices(base, np.s_[10:20, ::2])
    assert s1 == np.s_[20:40:2, 0, :4:2]
