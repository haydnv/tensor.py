import numpy as np

from sparse import SparseTensor


def test_eq():
    dims = [3]
    a = SparseTensor(dims)
    b = SparseTensor(dims)
    assert (a == b).all()

    dims = [5, 7, 1, 12]
    a = SparseTensor(dims)
    b = SparseTensor(dims)
    a[0, slice(1, -3, 2), :, slice(None, None, 4)] = 2
    b[0, slice(1, -3, 2), :, slice(None, None, 4)] = 2
    assert (a == b).all()


def test_setitem():
    dims = [7, 10]
    sparse = SparseTensor(dims)
    dense = np.zeros(dims)

    sparse_slice = sparse[slice(None, None, 2)][slice(None, -1, 3)]
    dense_slice = dense[slice(None, None, 2)][slice(None, -1, 3)]
    assert sparse_slice.shape == dense_slice.shape

    sparse[slice(None, None, 2)][slice(None, None, 3)] = 1
    dense[slice(None, None, 2)][slice(None, None, 3)] = 1
    assert (sparse.to_dense() == dense).all()

    dims = [5, 7, 3, 10]
    sparse = SparseTensor(dims)
    dense = np.zeros(dims)

    for coord in [(3, 0, 1, 9), (4, 0, 2, 1), (2,)]:
        sparse[coord] = 1
        dense[coord] = 1
    assert (sparse.to_dense() == dense).all()

    zero_out = (2, 0, slice(None), slice(1, -3, 3))
    sparse[zero_out] = 0
    dense[zero_out] = 0
    assert (sparse.to_dense() == dense).all()

    update = SparseTensor([3, 1])
    update[2, 0] = 1

    sparse[1] = update
    dense[1] = update.to_dense()
    assert (sparse.to_dense() == dense).all()

    sparse[:, :, slice(1, None, 2)][slice(1, -3, 3)] = 3
    dense[:, :, slice(1, None, 2)][slice(1, -3, 3)] = 3
    assert (sparse.to_dense() == dense).all()

    sparse[0, slice(None, None, 2)][slice(None, None, 3)] = 4
    dense[0, slice(None, None, 2)][slice(None, None, 3)] = 4
    sparse_slice = sparse[0, 0, :, :]
    dense_slice = dense[0, 0, :, :]
    assert (sparse_slice.to_dense() == dense_slice).all()
    assert (sparse.to_dense() == dense).all()


def test_getitem():
    dims = [3]
    sparse = SparseTensor(dims)
    dense = np.zeros(dims)

    sparse_slice = sparse[slice(2, 3, 2)]
    dense_slice = dense[slice(2, 3, 2)]
    assert sparse_slice.shape == dense_slice.shape

    sparse_slice = sparse[slice(1, 3)][slice(1, 2, 2)]
    dense_slice = dense[slice(1, 3)][slice(1, 2, 2)]
    assert sparse_slice.shape == dense_slice.shape

    dims = [5, 3, 1, 10]
    sparse = SparseTensor(dims)
    dense = np.zeros(dims)

    assert sparse[:].shape == dense[:].shape
    assert sparse[:, :].shape == dense[:, :].shape
    assert sparse[:, :, 0].shape == dense[:, :, 0].shape
    assert sparse[:, :, slice(1, None, 2)].shape == dense[:, :, slice(1, None, 2)].shape

    sparse_slice = sparse[slice(2), :][slice(1, None, 2)]
    dense_slice = dense[slice(2), :][slice(1, None, 2)]
    assert sparse_slice.shape == dense_slice.shape

    for coord in [(3, 1, 0, 9), (3, -1, 0, 1)]:
        sparse[coord] = 1
        dense[coord] = 1
        assert sparse[coord] == dense[coord]
        assert (sparse.to_dense() == dense).all()

    coord = (3, slice(1, 3))
    assert (sparse[coord].to_dense() == dense[coord]).all()


def test_broadcast():
    a = SparseTensor([2, 1, 3])
    b = SparseTensor([2, 3, 1])
    assert (a.broadcast(b.shape) == b.broadcast(a.shape)).all()

    ref_a = np.zeros([2, 1, 3])
    a[0] = 2
    ref_a[0] = 2

    ref_b = np.zeros([2, 3, 1])
    b[0] = 3
    ref_b[0] = 3

    a_b = a * b
    ref_a_b = ref_a * ref_b
    assert a_b.shape == ref_a_b.shape
    assert (a_b.to_dense() == ref_a_b).all()


def test_sum():
    dims = [3, 5, 2, 4]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)
    assert (sparse.sum(2).to_dense() == np.sum(ref, 2)).all()

    sparse[:, :, 0, slice(None, None, 3)] = 2
    ref[:, :, 0, slice(None, None, 3)] = 2

    for axis in range(4):
        assert (sparse.sum(axis).to_dense() == np.sum(ref, axis)).all()


def test_product():
    dims = [3, 5, 2, 4]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)
    assert (sparse.product(2).to_dense() == np.product(ref, 2)).all()

    sparse[:, :, 0, slice(None, None, 3)] = 2
    ref[:, :, 0, slice(None, None, 3)] = 2

    for axis in range(4):
        assert (sparse.product(axis).to_dense() == np.product(ref, axis)).all()


def test_expand_dims():
    dims = [3, 1, 5, 2]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    sparse[1, slice(None), slice(1, 5, 2), (0, 1)] = 1
    ref[1, slice(None), slice(1, 5, 2), (0, 1)] = 1
    assert (sparse.to_dense() == ref).all()

    for axis in [3, 1, 0]:
        sparse = sparse.expand_dims(axis)
        ref = np.expand_dims(ref, axis)
        assert (sparse.to_dense() == ref).all()

    sparse = sparse.expand_dims(sparse.ndim)
    ref = np.expand_dims(ref, ref.ndim)
    assert (sparse.to_dense() == ref).all()


def test_transpose():
    dims = [3, 2]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    sparse[slice(2), 1] = 1
    ref[slice(2), 1] = 1

    assert (sparse.transpose().to_dense() == ref.transpose()).all()
    assert (sparse.transpose([0, 1]).to_dense() == ref).all()
    assert (sparse.transpose([1, 0]).to_dense() == np.transpose(ref, [1, 0])).all()

    dims = [5, 1, 8, 3]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    sparse[:, :, slice(None, None, 2)] = 1
    ref[:, :, slice(None, None, 2)] = 1

    assert (sparse.transpose().to_dense() == ref.transpose()).all()
    assert (sparse.transpose([0, 2, 1, 3]).to_dense() == np.transpose(ref, [0, 2, 1, 3])).all()
    assert (sparse.transpose([3, 1, 2, 0]).to_dense() == np.transpose(ref, [3, 1, 2, 0])).all()


def test_multiply():
    sparse_a = SparseTensor([3])
    ref_a = np.zeros([3])
    sparse_b = SparseTensor([3])
    ref_b = np.zeros([3])

    sparse_a[0] = 3
    ref_a[0] = 3
    sparse_b[0] = 2
    ref_b[0] = 2
    assert ((sparse_a * sparse_b).to_dense() == ref_a * ref_b).all()

    sparse = SparseTensor([2, 5])
    dense = np.zeros([2, 5])
    sparse[0] = 1
    dense[0] = 1

    assert ((sparse * 5).to_dense() == (dense * 5)).all()


if __name__ == "__main__":
    test_eq()
    test_setitem()
    test_getitem()
    test_broadcast()
    test_sum()
    test_product()
    test_expand_dims()
    test_transpose()
    test_multiply()
    print("PASS")

