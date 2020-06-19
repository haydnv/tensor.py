import itertools
import numpy as np

from btree.btree import BTree
from collections.abc import Iterable


class SparseTensor(object):
    def from_dense(nparray):
        tensor = SparseTensor(nparray.shape, nparray.dtype)
        for coord in itertools.product(*[range(dim) for dim in nparray.shape]):
            tensor[coord] = nparray[coord]

        return tensor

    def __init__(self, shape, contents=[], dtype=np.int32):
        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            raise ValueError

        self.ndim = len(shape)
        self.shape = shape
        self.dtype = dtype

        values = (list(coord) + [val] for (coord, val) in contents)
        self.values = BTree(10, values)

    def __eq__(self, other):
        if self.shape != other.shape:
            return False

        for coord in itertools.product(*(range(dim) for dim in self.shape)):
            if self[coord] != other[coord]:
                return False

        return True

    def _validate_index(self, index):
        if len(index) > self.ndim:
            raise IndexError

        validated = []
        for i in range(self.ndim):
            if isinstance(index[i], slice):
                start = index[i].start if index[i].start else 0
                stop = index[i].stop if index[i].stop else self.shape[i]
                step = index[i].step if index[i].step else 1
                validated.append(slice(start, stop, step))
            elif isinstance(index[i], int):
                if abs(index[i]) > self.shape[i]:
                    raise IndexError
                else:
                    validated.append(index[i] if index[i] > 0 else self.shape[i] - index[i])

        return tuple(validated)

    def __getitem__(self, index):
        self._validate_index(index)

        if isinstance(index, int):
            index = (index,)

        if len(index) == 0:
            return []

        print(index)

        selected_dim = []
        for i in range(self.ndim):
            if i > len(index):
                selected_dim.append(self.shape[i])
            elif isinstance(index[i], slice):
                start = index[i].start if index[i].start else 0
                stop = index[i].end if index[i].end else self.shape[i]

                if start < 0:
                    start = self.shape[i] - start
                if stop < 0:
                    stop = self.shape[i] - stop

                if stop <= start:
                    break

                selected_dim.append(stop - start)
            elif isinstance(index[i], int):
                selected_dim.append(1)
            else:
                raise IndexError

        print("selection shape: {}".format(selected_dim))

        selected = []
        selection = ((translate_coord(index, v[:-1]), v[-1]) for v in self.values[index])
        return SparseTensor(selected_dim, selection)

    def __setitem__(self, index, value):
        index = self._validate_index(index)

        if value == 0:
            return

        if isinstance(index, int) or isinstance(index, slice):
            index = (index,)

        self.values.update(list(index + (value,)))

    def to_dense(self):
        arr = np.zeros(self.shape)
        for edge in self.values.select_all():
            arr[tuple(edge[:-1])] = edge[-1]

        return arr

    def __str__(self):
        return "{}".format(self.to_dense())

def _translate(index, source_coord):
    if not len(index) <= len(source_coord):
        raise IndexError

    target_coord = tuple(source_coord)
    for i in range(len(index)):
        if isinstance(index[i], slice):
            if source_coord[i] % index.step != 0:
                raise IndexError

            target_coord[i] = (source_coord[i] - index[i].start) // index.step
        else:
            target_coord[i] = source_coord[i] - index[i].start

    return target_coord

