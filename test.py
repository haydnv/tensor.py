import numpy as np

from tensor import SparseTensor


def test_setitem():
    dims = [5, 1, 3, 10]
    sparse = SparseTensor(dims)
    dense = np.zeros(dims)

    for coord in [(3, 0, 1, 9), (4, 0, 2, 1), (2,)]:
        sparse[coord] = 1
        dense[coord] = 1
    assert (sparse.to_dense() == dense).all()

    zero_out = (2, 0, 1)
    sparse[zero_out] = 0
    dense[zero_out] = 0
    assert (sparse.to_dense() == dense).all()

    update = SparseTensor([3, 1])
    update[2, 0] = 1

    sparse[1] = update
    dense[1] = update.to_dense()

    assert (sparse.to_dense() == dense).all()

    sparse[:, :, slice(None, None, 2)] = 3
    dense[:, :, slice(None, None, 2)] = 3
    assert (sparse.to_dense() == dense).all()

    sparse[:][0, 0, slice(None, None, 2)][slice(None, None, 3)] = 4
    dense[:][0, 0, slice(None, None, 2)][slice(None, None, 3)] = 4
    assert (sparse.to_dense() == dense).all()


def test_getitem():
    dims = [5, 3, 1, 10]
    sparse = SparseTensor(dims)
    dense = np.zeros(dims)

    assert sparse[:].shape == dense[:].shape
    assert sparse[:, :].shape == dense[:, :].shape
    assert sparse[:, :, 0].shape == dense[:, :, 0].shape
    assert sparse[:, :, slice(1, None, 2)].shape == dense[:, :, slice(1, None, 2)].shape

    for coord in [(3, 1, 0, 9), (3, -1, 0, 1)]:
        sparse[coord] = 1
        dense[coord] = 1
        assert sparse[coord] == dense[coord]
        assert (sparse.to_dense() == dense).all()

    coord = (3, slice(1, 3))
    assert (sparse[coord].to_dense() == dense[coord]).all()


if __name__ == "__main__":
    test_setitem()
    test_getitem()
    print("PASS")
