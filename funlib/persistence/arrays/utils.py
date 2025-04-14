from collections.abc import Sequence


def interleave(a: Sequence, b: Sequence, choices: Sequence[bool]) -> list:
    a_ind = iter(range(len([c for c in choices if c])))
    b_ind = iter(range(len([c for c in choices if not c])))
    return [a[next(a_ind)] if c else b[next(b_ind)] for c in choices]
