import numpy as np

from dense import BlockTensor
from sparse import SparseTensor


def test_eq():
    dims = [3]
    a = SparseTensor(dims)
    b = SparseTensor(dims)
    assert (a == b).all()

    a = BlockTensor(dims)
    b = BlockTensor(dims)
    assert (a == b).all()

    dims = [2, 1]
    sparse = SparseTensor(dims)
    dense = BlockTensor(dims)
    sparse[slice(1, None, 1)] = 3
    dense[slice(1, None, 1)] = 3
    assert (sparse[0].to_nparray() == sparse.to_nparray()[0]).all()
    assert (dense[0].to_nparray() == dense.to_nparray()[0]).all()
    assert (sparse == dense).all()

    dims = [5, 7, 1, 12]
    a = SparseTensor(dims)
    b = SparseTensor(dims)
    c = BlockTensor(dims)
    d = BlockTensor(dims)
    a[0, slice(1, -3, 2), :, slice(None, None, 4)] = 2
    b[0, slice(1, -3, 2), :, slice(None, None, 4)] = 2
    c[0, slice(1, -3, 2), :, slice(None, None, 4)] = 2
    d[0, slice(1, -3, 2), :, slice(None, None, 4)] = 2
    assert (a == b).all()
    assert (c == d).all()
    assert (a == c).all()


def test_setitem():
    dims = [7, 10]
    sparse = SparseTensor(dims)
    dense = BlockTensor(dims)
    ref = np.zeros(dims)

    sparse[0, 1] = 1
    dense[0, 1] = 1
    ref[0, 1] = 1
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()

    sparse_slice = sparse[slice(None, None, 2)][slice(None, -1, 3)]
    dense_slice = sparse[slice(None, None, 2)][slice(None, -1, 3)]
    ref_slice = ref[slice(None, None, 2)][slice(None, -1, 3)]
    assert sparse_slice.shape == ref_slice.shape
    assert dense_slice.shape == ref_slice.shape

    sparse[slice(None, None, 2)][slice(None, None, 3)] = 1
    dense[slice(None, None, 2)][slice(None, None, 3)] = 1
    ref[slice(None, None, 2)][slice(None, None, 3)] = 1
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()

    dims = [5, 7, 3, 10]
    sparse = SparseTensor(dims)
    dense = BlockTensor(dims)
    ref = np.zeros(dims)

    for coord in [(3, 0, 1, 9), (4, 0, 2, 1), (2,)]:
        sparse[coord] = 1
        dense[coord] = 1
        ref[coord] = 1
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()

    zero_out = (2, 0, slice(None), slice(1, -3, 3))
    sparse[zero_out] = 0
    dense[zero_out] = 0
    ref[zero_out] = 0
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()

    update = SparseTensor([3, 1])
    update[2, 0] = 1

    sparse[1] = update
    dense[1] = update
    ref[1] = update.to_nparray()
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()

    sparse[:, :, slice(1, None, 2)][slice(1, -3, 3)] = 3
    dense[:, :, slice(1, None, 2)][slice(1, -3, 3)] = 3
    ref[:, :, slice(1, None, 2)][slice(1, -3, 3)] = 3
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()

    sparse[0, slice(None, None, 2)][slice(None, None, 3)] = 4
    dense[0, slice(None, None, 2)][slice(None, None, 3)] = 4
    ref[0, slice(None, None, 2)][slice(None, None, 3)] = 4
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()

    sparse_slice = sparse[0]
    dense_slice = dense[0]
    ref_slice = ref[0]
    assert (sparse == dense).all()
    assert (sparse_slice.to_nparray() == ref_slice).all()
    assert (dense_slice.to_nparray() == ref_slice).all()

    sparse_slice = sparse[0, 0, :, :]
    dense_slice = dense[0, 0, :, :]
    ref_slice = ref[0, 0, :, :]
    assert (sparse_slice == dense_slice).all()
    assert (sparse_slice.to_nparray() == ref_slice).all()
    assert (dense_slice.to_nparray() == ref_slice).all()
    assert (sparse == dense).all()
    assert (sparse.to_nparray() == ref).all()
    assert (dense.to_nparray() == ref).all()


def test_getitem():
    dims = [3]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    sparse_slice = sparse[slice(2, 3, 2)]
    ref_slice = ref[slice(2, 3, 2)]
    assert sparse_slice.shape == ref_slice.shape

    sparse_slice = sparse[slice(1, 3)][slice(1, 2, 2)]
    ref_slice = ref[slice(1, 3)][slice(1, 2, 2)]
    assert sparse_slice.shape == ref_slice.shape

    dims = [5, 3, 1, 10]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    assert sparse[:].shape == ref[:].shape
    assert sparse[:, :].shape == ref[:, :].shape
    assert sparse[:, :, 0].shape == ref[:, :, 0].shape
    assert sparse[:, :, slice(1, None, 2)].shape == ref[:, :, slice(1, None, 2)].shape

    sparse_slice = sparse[slice(2), :][slice(1, None, 2)]
    ref_slice = ref[slice(2), :][slice(1, None, 2)]
    assert sparse_slice.shape == ref_slice.shape

    for coord in [(3, 1, 0, 9), (3, -1, 0, 1)]:
        sparse[coord] = 1
        ref[coord] = 1
        assert sparse[coord] == ref[coord]
        assert (sparse.to_nparray() == ref).all()

    coord = (3, slice(1, 3))
    assert (sparse[coord].to_nparray() == ref[coord]).all()


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
    assert (a_b.to_nparray() == ref_a_b).all()


def test_sum():
    dims = [3, 5, 2, 4]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)
    assert (sparse.sum(2).to_nparray() == np.sum(ref, 2)).all()

    sparse[:, :, 0, slice(None, None, 3)] = 2
    ref[:, :, 0, slice(None, None, 3)] = 2

    for axis in range(4):
        assert (sparse.sum(axis).to_nparray() == np.sum(ref, axis)).all()


def test_product():
    dims = [3, 5, 2, 4]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)
    assert (sparse.product(2).to_nparray() == np.product(ref, 2)).all()

    sparse[:, :, 0, slice(None, None, 3)] = 2
    ref[:, :, 0, slice(None, None, 3)] = 2

    for axis in range(4):
        assert (sparse.product(axis).to_nparray() == np.product(ref, axis)).all()


def test_expand_dims():
    dims = [3, 1, 5, 2]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    sparse[1, slice(None), slice(1, 5, 2), (0, 1)] = 1
    ref[1, slice(None), slice(1, 5, 2), (0, 1)] = 1
    assert (sparse.to_nparray() == ref).all()

    for axis in [3, 1, 0]:
        sparse = sparse.expand_dims(axis)
        ref = np.expand_dims(ref, axis)
        assert (sparse.to_nparray() == ref).all()

    sparse = sparse.expand_dims(sparse.ndim)
    ref = np.expand_dims(ref, ref.ndim)
    assert (sparse.to_nparray() == ref).all()


def test_transpose():
    dims = [3, 2]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    sparse[slice(2), 1] = 1
    ref[slice(2), 1] = 1

    assert (sparse.transpose().to_nparray() == ref.transpose()).all()
    assert (sparse.transpose([0, 1]).to_nparray() == ref).all()
    assert (sparse.transpose([1, 0]).to_nparray() == np.transpose(ref, [1, 0])).all()

    dims = [5, 1, 8, 3]
    sparse = SparseTensor(dims)
    ref = np.zeros(dims)

    sparse[:, :, slice(None, None, 2)] = 1
    ref[:, :, slice(None, None, 2)] = 1

    assert (sparse.transpose().to_nparray() == ref.transpose()).all()
    assert (sparse.transpose([0, 2, 1, 3]).to_nparray() == np.transpose(ref, [0, 2, 1, 3])).all()
    assert (sparse.transpose([3, 1, 2, 0]).to_nparray() == np.transpose(ref, [3, 1, 2, 0])).all()


def test_multiply():
    sparse_a = SparseTensor([3])
    ref_a = np.zeros([3])
    sparse_b = SparseTensor([3])
    ref_b = np.zeros([3])

    sparse_a[0] = 3
    ref_a[0] = 3
    sparse_b[0] = 2
    ref_b[0] = 2
    assert ((sparse_a * sparse_b).to_nparray() == ref_a * ref_b).all()

    sparse = SparseTensor([2, 5])
    dense = np.zeros([2, 5])
    sparse[0] = 1
    dense[0] = 1

    assert ((sparse * 5).to_nparray() == (dense * 5)).all()


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

