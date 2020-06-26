import numpy as np

from tensor import SparseTensor


def test_setitem():
    dims = [5, 1, 3, 10]
    sparse = SparseTensor(dims)
    dense = np.zeros(dims)

    for coord in [(3, 0, 1, 9), (4, 0, 2, 1), (2,)]:
        sparse[coord] = 1
        dense[coord] = 1

    zero_out = (2, 0, 1)
    sparse[zero_out] = 0
    dense[zero_out] = 0

    assert (sparse.to_dense() == dense).all()

if __name__ == "__main__":
    test_setitem()
    print("PASS")

