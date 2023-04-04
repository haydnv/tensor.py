import unittest

import numpy as np

from tensor import Block, Buffer, Tensor


class BufferTests(unittest.TestCase):
    def testAdd(self):
        x = Buffer(3, [1, 2, 3])
        y = Buffer(3, [4, 5, 6])
        self.assertTrue(all(x + y == Buffer(3, [5, 7, 9])))


class BlockTests(unittest.TestCase):
    def testMatMul(self):
        a_np = np.arange(6).reshape((2, 3))
        a_tc = Block((2, 3), range(6))
        b_np = np.arange(12).reshape((3, 4))
        b_tc = Block((3, 4), range(12))

        expected = a_np @ b_np
        actual = a_tc @ b_tc

        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(all(e == a for e, a in zip(expected.flatten(), actual)))

    def testMatMulBatch(self):
        a_np = np.arange(18).reshape((3, 2, 3))
        a_tc = Block((3, 2, 3), range(18))
        b_np = np.arange(36).reshape((3, 3, 4))
        b_tc = Block((3, 3, 4), range(36))

        expected = a_np @ b_np
        actual = a_tc @ b_tc

        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(all(e == a for e, a in zip(expected.flatten(), actual)))

    def testMul(self):
        x_np = np.arange(12).reshape((3, 1, 4))
        x_tc = Block((3, 1, 4), range(12))

        y_np = np.arange(4).reshape(4)
        y_tc = Block((4,), range(4))

        expected = x_np * y_np
        actual = x_tc * y_tc

        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(all(e == a for e, a in zip(expected.flatten(), actual)))

    def testSlice(self):
        shape = [2, 3, 4, 5, 6]

        x_np = np.arange(np.product(shape)).reshape(shape)
        x_tc = Block(shape, [int(n) for n in x_np.flatten()])

        expected = x_np[1, :, 1:, :-2]
        actual = x_tc[1, :, 1:, :-2]

        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(all(e == a for e, a in zip(expected.flatten(), actual)))

    def testSum(self):
        shape = [2, 3, 4, 5, 6]

        x_np = np.arange(np.product(shape)).reshape(shape)
        x_tc = Block(shape, [int(n) for n in x_np.flatten()])

        expected = x_np.sum(3).sum(1)
        actual = x_tc.reduce_sum([1, 3])

        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(all(e == a for e, a in zip(expected.flatten(), actual)))


class TensorTests(unittest.TestCase):
    def testAdd(self):
        x = Tensor((2, 3), [[0, 1, 2, 3, 4, 5]])
        y = Tensor((2, 3), [[5, 4, 3, 2, 1, 0]])
        self.assertTrue(all(x + y == Tensor((2, 3), [[5, 5, 5, 5, 5, 5]])))

    def testMul(self):
        x = Tensor((2, 1), [[0, 1]])
        y = Tensor((2, 1), [[5, 4]])
        self.assertTrue(all(x + y == Tensor((2, 1), [[0, 4]])))


if __name__ == "__main__":
    unittest.main()
