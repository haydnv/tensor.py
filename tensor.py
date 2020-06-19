import functools
import numpy as np

from btree.btree import BTree


class SparseTensor(object):
    def __init__(self, dims, dtype=np.int32):
        if dims != list(dims) or len(dims) > 2:
            raise ValueError

        self.dims = dims
        self.dtype = dtype
        self.values = BTree(10)

    def expand(self, new_dims):
        if new_dims != list(new_dims) or len(new_dims) > 2 or len(new_dims) < len(self.dims):
            raise ValueError

        for i in range(len(self.dims)):
            if new_dims[i] < self.dims[i]:
                raise ValueError

        self.dims = new_dims

    def __setitem__(self, index, value):
        if len(self.dims) == 1:
            index = (index,)

        if len(index) != len(self.dims):
            raise ValueError

        if [value] != np.array([value]).astype(self.dtype):
            raise ValueError

        self.values.insert(list(index + (value,)))

    def dense(self):
        arr = np.zeros(self.dims)
        for edge in self.values.select_all():
            arr[tuple(edge[:-1])] = edge[-1]

        return arr

    def transpose(self):
        raise NotImplementedError

