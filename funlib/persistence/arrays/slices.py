import numpy as np


def chain_slices(slices_a, slices_b):

    # make sure both slice expressions are tuples
    if not isinstance(slices_a, tuple):
        slices_a = (slices_a,)
    if not isinstance(slices_b, tuple):
        slices_b = (slices_b,)

    # dimension of a is number of non-int expressions
    dim_a = sum([not isinstance(x, int) for x in slices_a])

    # slices_b can't slice more dimensions than a has
    assert (
        len(slices_b) <= dim_a
    ), f"Slice expression {slices_b} has too many dimensions to chain with {slices_a}"

    chained = []

    j = 0
    for slice_a in slices_a:

        # if slice_a is int that dimension does not exist any longer, skip
        # also skip if b has no more elements
        if j == len(slices_b) or isinstance(slice_a, int):
            chained.append(slice_a)
        else:
            slice_b = slices_b[j]
            chained.append(_chain_slice(slice_a, slice_b))
            j += 1

    return tuple(chained)


def _chain_slice(a, b):

    # a is a slice(start, stop, step) expression
    if isinstance(a, slice):

        start_a = a.start if a.start else 0
        step_a = a.step if a.step else 1

        if isinstance(b, int):

            idx = start_a + step_a * b
            assert not a.stop or idx < a.stop, f"Slice {b} out of range for {b}"
            return idx

        elif isinstance(b, slice):

            start_b = b.start if b.start else 0
            step_b = b.step if b.step else 1

            start = start_a + step_a * start_b if a.start or b.start else None
            stop = step_a * b.stop if b.stop else a.stop
            step = step_a * step_b if a.step or b.step else None

            return slice(start, stop, step)

        elif isinstance(b, list):

            return list(_chain_slice(a, x) for x in b)

        elif isinstance(b, np.ndarray):

            # is b a mask array?
            if b.dtype == bool:
                raise RuntimeError("Not yet implemented")

            return np.array([_chain_slice(a, x) for x in b])

        else:

            raise RuntimeError(
                f"Don't know how to deal with slice {b} of type {type(b)}"
            )

    # is an index array
    elif isinstance(a, list):

        return list(np.array(a)[(b,)])

    elif isinstance(a, np.ndarray):

        if a.dtype == bool:
            raise RuntimeError("Not yet implemented")

        return a[(b,)]

    else:

        raise RuntimeError(f"Don't know how to deal with slice {a} of type {type(a)}")
