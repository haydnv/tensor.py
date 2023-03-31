import unittest

from tensor import Buffer, Tensor


class BufferTests(unittest.TestCase):
    def testAdd(self):
        x = Buffer(3, [1, 2, 3])
        y = Buffer(3, [4, 5, 6])
        self.assertTrue(all(x + y == Buffer(3, [5, 7, 9])))


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
