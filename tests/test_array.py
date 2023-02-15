from funlib.persistence.arrays import Array
from funlib.geometry import Roi, Coordinate
import numpy as np
import unittest


class TestArray(unittest.TestCase):
    def test_constructor(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        roi = Roi((0, 0, 0), (10, 10, 10))

        # consistent configurations
        Array(data, roi, (1, 1, 1))
        Array(data, roi, (1, 1, 2))
        Array(data, roi, (1, 5, 2))
        Array(data, roi, (10, 5, 2))
        roi = Roi((1, 1, 1), (10, 10, 10))
        Array(data, roi, (1, 1, 1), data_offset=(1, 1, 1))
        roi = Roi((0, 0, 0), (20, 20, 20))
        Array(data, roi, (2, 2, 2))

        # dims don't match
        with self.assertRaises(AssertionError):
            Array(data, roi, (1, 1))

        # ROI not multiple of voxel size
        with self.assertRaises(AssertionError):
            Array(data, roi, (1, 1, 3))
        with self.assertRaises(AssertionError):
            Array(data, roi, (1, 1, 4))

        # ROI begin doesn't align with voxel size
        roi = Roi((1, 1, 1), (11, 11, 11))
        with self.assertRaises(AssertionError):
            Array(data, roi, (1, 1, 2))

        # ROI shape doesn't align with voxel size
        roi = Roi((0, 0, 0), (11, 11, 11))
        with self.assertRaises(AssertionError):
            Array(data, roi, (1, 1, 2))

        # ROI outside of provided data
        roi = Roi((0, 0, 0), (20, 20, 20))
        with self.assertRaises(AssertionError):
            Array(data, roi, (1, 1, 1))
        with self.assertRaises(AssertionError):
            Array(data, roi, (2, 2, 1))
        with self.assertRaises(AssertionError):
            Array(data, roi, (2, 2, 2), data_offset=(0, 0, 2))

    def test_shape(self):
        # ROI fits data

        a1 = Array(np.zeros((10,)), Roi((0,), (10,)), (1,))
        a2 = Array(np.zeros((10, 10)), Roi((0, 0), (10, 10)), (1, 1))
        a2_3 = Array(np.zeros((3, 10, 10)), Roi((0, 0), (10, 10)), (1, 1))
        a5_3_2_1 = Array(
            np.zeros((1, 2, 3, 4, 4, 4, 4, 4)),
            Roi((0, 0, 0, 0, 0), (80, 80, 80, 80, 80)),
            (20, 20, 20, 20, 20),
        )

        assert a1.shape == (10,)
        assert a2.shape == (10, 10)
        assert a2_3.shape == (3, 10, 10)
        assert a2_3.roi.dims == 2
        assert a5_3_2_1.shape == (1, 2, 3, 4, 4, 4, 4, 4)

        # ROI subset of data

        a1 = Array(np.zeros((20,)), Roi((0,), (10,)), (1,))
        a2 = Array(np.zeros((20, 20)), Roi((0, 0), (10, 10)), (1, 1))
        a2_3 = Array(np.zeros((3, 20, 20)), Roi((0, 0), (10, 10)), (1, 1))
        a5_3_2_1 = Array(
            np.zeros((1, 2, 3, 5, 5, 5, 5, 5)),
            Roi((0, 0, 0, 0, 0), (80, 80, 80, 80, 80)),
            (20, 20, 20, 20, 20),
        )

        assert a1.shape == (10,)
        assert a2.shape == (10, 10)
        assert a2_3.shape == (3, 10, 10)
        assert a2_3.roi.dims == 2
        assert a5_3_2_1.shape == (1, 2, 3, 4, 4, 4, 4, 4)

    def test_dtype(self):
        for dtype in [np.float32, np.uint8, np.uint64]:
            assert (
                Array(np.zeros((1,), dtype=dtype), Roi((0,), (1,)), (1,)).dtype == dtype
            )

    def test_getitem(self):
        a = Array(np.arange(0, 10).reshape(2, 5), Roi((0, 0), (2, 5)), (1, 1))

        assert a[Coordinate((0, 0))] == 0
        assert a[Coordinate((0, 1))] == 1
        assert a[Coordinate((0, 2))] == 2
        assert a[Coordinate((1, 0))] == 5
        assert a[Coordinate((1, 1))] == 6
        with self.assertRaises(AssertionError):
            a[Coordinate((1, 5))]
        with self.assertRaises(AssertionError):
            a[Coordinate((2, 5))]
        with self.assertRaises(AssertionError):
            a[Coordinate((-1, 0))]
        with self.assertRaises(AssertionError):
            a[Coordinate((0, -1))]

        b = a[Roi((1, 1), (1, 4))]
        with self.assertRaises(AssertionError):
            b[Coordinate((0, 0))] == 0
        with self.assertRaises(AssertionError):
            b[Coordinate((0, 1))] == 1
        with self.assertRaises(AssertionError):
            b[Coordinate((0, 2))] == 2
        with self.assertRaises(AssertionError):
            b[Coordinate((1, 0))] == 5
        assert b[Coordinate((1, 1))] == 6
        assert b[Coordinate((1, 2))] == 7
        assert b[Coordinate((1, 3))] == 8
        assert b[Coordinate((1, 4))] == 9
        with self.assertRaises(AssertionError):
            b[Coordinate((1, 5))]
        with self.assertRaises(AssertionError):
            b[Coordinate((2, 5))]
        with self.assertRaises(AssertionError):
            b[Coordinate((-1, 0))]
        with self.assertRaises(AssertionError):
            b[Coordinate((0, -1))]

    def test_setitem(self):
        # set entirely with numpy array

        a = Array(np.zeros((2, 5)), Roi((0, 0), (2, 5)), (1, 1))

        a[Roi((0, 0), (2, 5))] = np.arange(0, 10).reshape(2, 5)
        assert a[Coordinate((0, 0))] == 0
        assert a[Coordinate((0, 1))] == 1
        assert a[Coordinate((0, 2))] == 2
        assert a[Coordinate((1, 0))] == 5
        assert a[Coordinate((1, 1))] == 6
        assert a[Coordinate((1, 4))] == 9

        # set entirely with numpy array and channels

        a = Array(np.zeros((3, 2, 5)), Roi((0, 0), (2, 5)), (1, 1))

        a[Roi((0, 0), (2, 5))] = np.arange(0, 3 * 10).reshape(3, 2, 5)
        np.testing.assert_array_equal(a[Coordinate((0, 0))], [0, 10, 20])
        np.testing.assert_array_equal(a[Coordinate((0, 1))], [1, 11, 21])
        np.testing.assert_array_equal(a[Coordinate((1, 4))], [9, 19, 29])

        # set entirely with scalar

        a = Array(np.zeros((2, 5)), Roi((0, 0), (2, 5)), (1, 1))

        a[Roi((0, 0), (2, 5))] = 42
        assert a[Coordinate((0, 0))] == 42
        assert a[Coordinate((1, 4))] == 42

        # set partially with scalar and channels

        a = Array(np.arange(0, 3 * 10).reshape(3, 2, 5), Roi((0, 0), (2, 5)), (1, 1))

        a[Roi((0, 0), (2, 2))] = 42
        np.testing.assert_array_equal(a[Coordinate((0, 0))], [42, 42, 42])
        np.testing.assert_array_equal(a[Coordinate((0, 1))], [42, 42, 42])
        np.testing.assert_array_equal(a[Coordinate((0, 2))], [2, 12, 22])
        np.testing.assert_array_equal(a[Coordinate((1, 2))], [7, 17, 27])
        np.testing.assert_array_equal(a[Coordinate((1, 3))], [8, 18, 28])
        np.testing.assert_array_equal(a[Coordinate((1, 4))], [9, 19, 29])

        # set partially with Array

        a = Array(np.zeros((2, 5)), Roi((0, 0), (2, 5)), (1, 1))
        b = Array(np.arange(0, 10).reshape(2, 5), Roi((0, 0), (2, 5)), (1, 1))

        a[Roi((0, 0), (1, 5))] = b[Roi((0, 0), (1, 5))]
        assert a[Coordinate((0, 0))] == 0
        assert a[Coordinate((0, 1))] == 1
        assert a[Coordinate((0, 2))] == 2
        assert a[Coordinate((1, 0))] == 0
        assert a[Coordinate((1, 1))] == 0
        assert a[Coordinate((1, 4))] == 0

        a = Array(np.zeros((2, 5)), Roi((0, 0), (2, 5)), (1, 1))
        b = Array(np.arange(0, 10).reshape(2, 5), Roi((0, 0), (2, 5)), (1, 1))

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

    def test_materialize(self):
        a = Array(np.arange(0, 10).reshape(2, 5), Roi((0, 0), (2, 5)), (1, 1))

        b = a[Roi((0, 0), (2, 2))]

        # underlying data did not change
        assert a.data.shape == b.data.shape

        assert b.shape == (2, 2)
        b.materialize()
        assert b.shape == (2, 2)

        assert b.data.shape == (2, 2)

    def test_to_ndarray(self):
        a = Array(np.arange(0, 10).reshape(2, 5), Roi((0, 0), (2, 5)), (1, 1))

        # not within ROI of a and no fill value provided
        with self.assertRaises(AssertionError):
            a.to_ndarray(Roi((0, 0), (5, 5)))

        b = a.to_ndarray(Roi((0, 0), (1, 5)))
        compare = np.array([[0, 1, 2, 3, 4]])

        b = a.to_ndarray(Roi((1, 0), (1, 5)))
        compare = np.array([[5, 6, 7, 8, 9]])

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

    def test_intersect(self):
        a = Array(np.arange(0, 10).reshape(2, 5), Roi((0, 0), (2, 5)), (1, 1))

        b = a.intersect(Roi((1, 1), (10, 10)))

        assert b.roi == Roi((1, 1), (1, 4))
        np.testing.assert_array_equal(b.to_ndarray(), [[6, 7, 8, 9]])
