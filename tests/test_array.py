import dask.array as da
import numpy as np
import pytest

from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import Array


@pytest.mark.parametrize("array_constructor", [np.zeros, da.zeros])
def test_constructor(array_constructor):
    data = array_constructor((10, 10, 10), dtype=np.float32)
    offset = Coordinate(0, 0, 0)

    # consistent configurations
    Array(data, offset, (1, 1, 1))
    Array(data, offset, (1, 1, 2))
    Array(data, offset, (1, 5, 2))
    Array(data, offset, (10, 5, 2))
    Array(data, (1, 1, 1), (1, 1, 1))
    Array(data, offset, (2, 2, 2))

    # dims don't match
    with pytest.raises(ValueError):
        Array(data, offset, (1, 1))

    Array(data, offset, (1, 1, 3))
    Array(data, offset, (1, 1, 4))

    Array(data, (1, 1, 1), (1, 1, 2))


def test_dtype():
    for dtype in [np.float32, np.uint8, np.uint64]:
        assert Array(np.zeros((1,), dtype=dtype), (0,), (1,)).dtype == dtype


def test_getitem():
    a = Array(np.arange(0, 10).reshape(2, 5), (0, 0), (1, 1))

    assert a[Coordinate((0, 0))] == 0
    assert a[0, 0] == 0
    assert a[Coordinate((0, 1))] == 1
    assert a[0, 1] == 1
    assert a[Coordinate((0, 2))] == 2
    assert a[0, 2] == 2
    assert a[Coordinate((1, 0))] == 5
    assert a[1, 0] == 5
    assert a[Coordinate((1, 1))] == 6
    assert a[1, 1] == 6
    with pytest.raises(IndexError):
        a[Coordinate((1, 5))]
    with pytest.raises(IndexError):
        a[1, 5]
    with pytest.raises(IndexError):
        a[Coordinate((2, 5))]
    with pytest.raises(IndexError):
        a[2, 5]
    with pytest.raises(IndexError):
        a[Coordinate((-1, 0))]
    with pytest.raises(IndexError):
        a[Coordinate((0, -1))]

    # Test negative indexes
    assert a[0, -1] == 4
    assert a[-1, 0] == 5

    b = a[Roi((1, 1), (1, 4))]
    np.testing.assert_array_equal(b, [[6, 7, 8, 9]])


def test_setitem():
    # set entirely with numpy array

    a = Array(np.zeros((2, 5)), (0, 0), (1, 1))

    data = np.arange(0, 10).reshape(2, 5)
    a[Roi((0, 0), (2, 5))] = data
    assert a[Coordinate((0, 0))] == a._source_data[0, 0] == 0
    assert a[Coordinate((0, 1))] == a._source_data[0, 1] == 1
    assert a[Coordinate((0, 2))] == a._source_data[0, 2] == 2
    assert a[Coordinate((1, 0))] == a._source_data[1, 0] == 5
    assert a[Coordinate((1, 1))] == a._source_data[1, 1] == 6
    assert a[Coordinate((1, 4))] == a._source_data[1, 4] == 9

    # set entirely with numpy array and channels

    a = Array(np.zeros((3, 2, 5)), (0, 0), (1, 1))

    a[Roi((0, 0), (2, 5))] = np.arange(0, 3 * 10).reshape(3, 2, 5)
    np.testing.assert_array_equal(a[Coordinate((0, 0))], [0, 10, 20])
    np.testing.assert_array_equal(a[Coordinate((0, 1))], [1, 11, 21])
    np.testing.assert_array_equal(a[Coordinate((1, 4))], [9, 19, 29])

    # set entirely with scalar

    a = Array(np.zeros((2, 5)), (0, 0), (1, 1))

    a[Roi((0, 0), (2, 5))] = 42
    assert a[Coordinate((0, 0))] == 42
    assert a[Coordinate((1, 4))] == 42

    # set partially with scalar and channels

    a = Array(np.arange(0, 3 * 10).reshape(3, 2, 5), (0, 0), (1, 1))

    a[Roi((0, 0), (2, 2))] = 42
    np.testing.assert_array_equal(a[Coordinate((0, 0))], [42, 42, 42])
    np.testing.assert_array_equal(a[Coordinate((0, 1))], [42, 42, 42])
    np.testing.assert_array_equal(a[Coordinate((0, 2))], [2, 12, 22])
    np.testing.assert_array_equal(a[Coordinate((1, 2))], [7, 17, 27])
    np.testing.assert_array_equal(a[Coordinate((1, 3))], [8, 18, 28])
    np.testing.assert_array_equal(a[Coordinate((1, 4))], [9, 19, 29])

    # set partially with Array

    a = Array(np.zeros((2, 5)), (0, 0), (1, 1))
    b = Array(np.arange(0, 10).reshape(2, 5), (0, 0), (1, 1))

    a[Roi((0, 0), (1, 5))] = b[Roi((0, 0), (1, 5))]
    assert a[Coordinate((0, 0))] == 0
    assert a[Coordinate((0, 1))] == 1
    assert a[Coordinate((0, 2))] == 2
    assert a[Coordinate((1, 0))] == 0
    assert a[Coordinate((1, 1))] == 0
    assert a[Coordinate((1, 4))] == 0

    a = Array(np.zeros((2, 5)), (0, 0), (1, 1))
    b = Array(np.arange(0, 10).reshape(2, 5), (0, 0), (1, 1))

    a[Roi((0, 0), (1, 5))] = b[Roi((1, 0), (1, 5))]
    assert a[Coordinate((0, 0))] == 5
    assert a[Coordinate((0, 1))] == 6
    assert a[Coordinate((0, 4))] == 9
    assert a[Coordinate((1, 0))] == 0
    assert a[Coordinate((1, 1))] == 0
    assert a[Coordinate((1, 2))] == 0

    a[Roi((1, 0), (1, 5))] = b[Roi((0, 0), (1, 5))]
    assert a[Coordinate((0, 0))] == 5
    assert a[Coordinate((0, 1))] == 6
    assert a[Coordinate((0, 4))] == 9
    assert a[Coordinate((1, 0))] == 0
    assert a[Coordinate((1, 1))] == 1
    assert a[Coordinate((1, 2))] == 2


def test_to_ndarray():
    a = Array(np.arange(0, 10).reshape(2, 5), (0, 0), (1, 1))

    b = a.to_ndarray(Roi((0, 0), (1, 5)))
    compare = np.array([[0, 1, 2, 3, 4]])
    np.testing.assert_array_equal(b, compare)

    b = a.to_ndarray(Roi((1, 0), (1, 5)))
    compare = np.array([[5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(b, compare)

    b = a.to_ndarray(Roi((0, 0), (2, 2)))
    compare = np.array([[0, 1], [5, 6]])
    np.testing.assert_array_equal(b, compare)

    b = a.to_ndarray(Roi((0, 0), (5, 5)), fill_value=1)
    compare = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(b, compare)


def test_to_ndarray_with_slices():
    a = Array(
        np.arange(0, 10 * 10).reshape(10, 2, 5), (0, 0), (1, 1), lazy_op=slice(0, 1)
    )

    b = a.to_ndarray(Roi((0, 0), (1, 5)))
    compare = np.array([[[0, 1, 2, 3, 4]]])
    np.testing.assert_array_equal(b, compare)

    b = a.to_ndarray(Roi((1, 0), (1, 5)))
    compare = np.array([[[5, 6, 7, 8, 9]]])
    np.testing.assert_array_equal(b, compare)

    b = a.to_ndarray(Roi((0, 0), (2, 2)))
    compare = np.array([[[0, 1], [5, 6]]])
    np.testing.assert_array_equal(b, compare)

    b = a.to_ndarray(Roi((0, 0), (5, 5)), fill_value=1)
    compare = np.array(
        [
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ]
    )
    np.testing.assert_array_equal(b, compare)


def test_adapters():
    a = Array(np.arange(0, 10 * 10).reshape(10, 2, 5), (0, 0), (1, 1))
    assert a.dtype == int

    a = Array(
        np.arange(0, 10 * 10).reshape(10, 2, 5), (0, 0), (1, 1), lazy_op=lambda x: x > 2
    )
    assert a.dtype == bool

    a = Array(
        np.arange(0, 10 * 10).reshape(10, 2, 5),
        (0, 0),
        (1, 1),
        lazy_op=lambda x: x + 0.5,
    )
    assert a.dtype == float


def test_slicing():
    a = Array(np.arange(0, 4 * 4).reshape(4, 2, 2), (0, 0), (1, 1))

    a.lazy_op(np.s_[0:3, 1, :])
    assert a.shape == (3, 2)
    assert a.axis_names == ["c0^", "d1"], a.axis_names
    assert a.units == [""]

    a.lazy_op(np.s_[2, :])
    assert a.shape == (2,)
    assert a.axis_names == ["d1"]
    assert a.units == [""]

    a[:] = 42

    assert all([x == 42 for x in a._source_data[2, 1, :]]), a._source_data[2, 1, :]

    # test with list indexing
    a = Array(np.arange(0, 4 * 4).reshape(4, 2, 2), (0, 0), (1, 1))

    a.lazy_op(np.s_[[0, 1, 2], 1, :])
    assert a.shape == (3, 2)
    assert a.axis_names == ["c0^", "d1"]
    assert a.units == [""]

    a.lazy_op(np.s_[2, :])
    assert a.shape == (2,)
    assert a.axis_names == ["d1"]
    assert a.units == [""]

    a[:] = 42

    assert all([x == 42 for x in a._source_data[2, 1, :]]), a._source_data[:]

    # test weird case
    a = Array(np.arange(0, 4 * 4).reshape(4, 2, 2), (0, 0), (1, 1))

    a.lazy_op(np.s_[[2, 2, 2], 1, :])
    assert a.shape == (3, 2)
    assert a.axis_names == ["c0^", "d1"]
    assert a.units == [""]

    a[:, :] = np.array([42, 43, 44]).reshape(3, 1)
    assert all([x == 44 for x in a._source_data[2, 1, :]]), a._source_data[2, 1, :]

    # test_bool_indexing
    a = Array(np.arange(0, 4 * 4).reshape(4, 2, 2), (0, 0), (1, 1))

    a.lazy_op(np.s_[np.array([True, True, True, False]), 1, :])
    assert a.shape == (3, 2)
    assert a.axis_names == ["c0^", "d1"]
    assert a.units == [""]

    with pytest.raises(RuntimeError):
        a[:, :] = np.array([42, 43, 44]).reshape(3, 1)


def test_slicing_channel_dim_last():
    a = Array(
        np.arange(0, 4 * 4).reshape(2, 2, 4),
        (0, 0),
        (1, 1),
        types=["space", "time", "channel"],
    )
    assert a.roi == Roi((0, 0), (2, 2))

    a.lazy_op(np.s_[1, :, 0:3])
    assert a.roi == Roi((0,), (2,))
    assert a.shape == (2, 3)
    assert a.axis_names == ["d1", "c0^"], a.axis_names
    assert a.types == ["time", "channel"], a.types
    assert a.units == [""]

    a.lazy_op(np.s_[:, 2])
    assert a.roi == Roi((0,), (2,))
    assert a.shape == (2,)
    assert a.axis_names == ["d1"]
    assert a.types == ["time"]
    assert a.units == [""]

    a[:] = 42

    assert all([x == 42 for x in a._source_data[1, :, 2]]), a._source_data[1, :, 2]

    # test with list indexing
    a = Array(
        np.arange(0, 4 * 4).reshape(2, 2, 4),
        (0, 0),
        (1, 1),
        axis_names=["d0", "d1", "c0^"],
    )

    a.lazy_op(np.s_[[0, 1], 1, :])
    assert a.roi == Roi((0,), (2,))
    assert a.shape == (2, 4)
    assert a.axis_names == ["d0", "c0^"]
    assert a.types == ["space", "channel"]
    assert a.units == [""]

    a.lazy_op(np.s_[1, :])
    assert a.roi == Roi(tuple(), tuple())
    assert a.shape == (4,)
    assert a.axis_names == ["c0^"]
    assert a.types == ["channel"]
    assert a.units == []

    a[:] = 42

    assert all([x == 42 for x in a._source_data[1, 1, :]]), a._source_data[1, 1, :]

    # test weird case
    a = Array(
        np.arange(0, 4 * 4).reshape(4, 2, 2),
        (0, 0),
        (1, 1),
        types=["time", "space", "channel"],
    )

    a.lazy_op(np.s_[[2, 2, 2], 1, :])
    # assert a.roi == None  # TODO: This doesn't make sense???
    assert a.shape == (3, 2)
    assert a.axis_names == ["d0", "c0^"]
    assert a.types == ["time", "channel"]
    assert a.units == [""]

    a[:, :] = np.array([42, 43, 44]).reshape(3, 1)
    assert all([x == 44 for x in a._source_data[2, 1, :]]), a._source_data[2, 1, :]

    # test_bool_indexing
    a = Array(
        np.arange(0, 4 * 4).reshape(2, 2, 4),
        (0, 0),
        (1, 1),
        axis_names=["d0", "d1", "c0^"],
        types=["time", "space", "channel"],
    )

    a.lazy_op(np.s_[1, :, np.array([True, True, True, False])])
    assert a.roi == Roi((0,), (2,))
    assert a.shape == (2, 3)
    assert a.axis_names == ["d1", "c0^"]
    assert a.types == ["space", "channel"]
    assert a.units == [""]

    with pytest.raises(RuntimeError):
        a[:, :] = np.array([42, 43, 44]).reshape(3, 1)


def test_lazy_op():
    a = Array(
        np.arange(0, 4 * 4).reshape(4, 2, 2),
        (0, 0),
        (1, 1),
        types=["space", "channel", "time"],
    )
    assert a.roi == Roi((0, 0), (4, 2))
    assert a.shape == (4, 2, 2)

    a.lazy_op(Roi((2, 0), (2, 2)))
    assert a.roi == Roi((2, 0), (2, 2))
    assert a.shape == (2, 2, 2)
    assert a.axis_names == ["d0", "c0^", "d1"]

    a.lazy_op(np.s_[:, 0, :])
    assert a.roi == Roi((2, 0), (2, 2))
    assert a.shape == (2, 2)
    assert a.axis_names == ["d0", "d1"]

    a.lazy_op(Roi((2, 0), (1, 2)))
    assert a.roi == Roi((2, 0), (1, 2))
    assert a.shape == (1, 2)

    assert a.dtype == int
    a.lazy_op(lambda b: b > 2)
    assert a.dtype == bool


def test_writeable():
    a = Array(
        np.arange(0, 4 * 4).reshape(4, 2, 2),
        (0, 0),
        (1, 1),
        types=["space", "channel", "time"],
    )
    assert a.is_writeable

    a.lazy_op(np.s_[0:3, 1, :])
    assert a.shape == (3, 2)
    assert a.axis_names == ["d0", "d1"]
    assert a.is_writeable

    a.lazy_op(Roi((0, 0), (2, 2)))
    assert a.shape == (2, 2)
    assert a.axis_names == ["d0", "d1"]
    assert a.is_writeable

    a.lazy_op(lambda b: b > 2)
    assert a.shape == (2, 2)
    assert a.axis_names == ["d0", "d1"]
    assert not a.is_writeable
