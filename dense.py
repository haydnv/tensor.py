import itertools
import math
import numpy as np
import sys

from base import affected, product, validate_match
from transform import SliceRebase

BLOCK_SIZE = 1000
BLOCK_OF_COORDS_LEN = BLOCK_SIZE // sys.getsizeof(np.uint64())


def shape_of(source_shape, match):
    shape = []

    for axis in range(len(match)):
        assert match[axis] is not None
        if isinstance(match[axis], slice):
            s = match[axis]
            assert s.start is not None and s.start >= 0 and s.stop is not None and s.stop >= 0
            shape.append(math.ceil((s.stop - s.start) / s.step))
        elif isinstance(match[axis], tuple):
            assert (np.array(match[axis]) >= 0).all()
            shape.append(len(match[axis]))

    for axis in range(len(match), len(source_shape)):
        shape.append(source_shape[axis])

    return tuple(shape)


class BlockList(object):
    def __init__(self, shape, dtype):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
        self.size = product(shape)
        self._per_block = BLOCK_SIZE // sys.getsizeof(dtype())
        self._blocks = [np.zeros([self._per_block])] * (self.size // self._per_block)
        if self.size % self._per_block > 0:
            self._blocks.append(np.zeros([self.size % self._per_block]))

        self._coord_index = np.array([product(shape[axis + 1:]) for axis in range(self.ndim)])

    def __getitem__(self, match):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            index = sum(np.array(match) * self._coord_index)
            return self._blocks[index // self._per_block][index % self._per_block]
        else:
            return BlockListSlice(self, match)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            index = sum(np.array(match) * self._coord_index)
            self._blocks[index // self._per_block][index % self._per_block] = self.dtype(value)
        else:
            affected_range = itertools.product(*affected(match, self.shape))
            value = self.dtype(value)
            for coords in chunk_iter(affected_range, BLOCK_OF_COORDS_LEN):
                coords = [list(coord) for coord in coords]
                block = np.array(coords) * self._coord_index
                block = block.sum(1)

                indices = block // self._per_block
                offsets = block % self._per_block

                i = 0
                for block_id in np.unique(indices):
                    num_to_update = np.sum(indices == block_id)
                    self._blocks[block_id][offsets[i:(i + num_to_update)]] = value
                    i += num_to_update


class BlockListSlice(object):
    def __init__(self, source, match):
        self._source = source
        self._rebase = SliceRebase(source.shape, match)

    def __getitem__(self, match):
        return self._source[self._rebase.invert_coord(match)]

    def __setitem__(self, match, value):
        self._source[self._rebase.invert_coord(match)] = value


class DenseTensor(object):
    def __init__(self, shape, dtype=np.int32, block_list=None):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
        self.size = product(shape)

        if block_list is None:
            self._block_list = BlockList(shape, dtype)
        else:
            self._block_list = block_list

    def __getitem__(self, match):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            return self._block_list[match]
        else:
            block_list = self._block_list[match]
            return DenseTensor(shape_of(self.shape, match), self.dtype, block_list)

    def to_nparray(self):
        arr = np.zeros(self.shape, self.dtype)
        for coord in itertools.product(*[range(dim) for dim in self.shape]):
            arr[coord] = self[coord]
        return arr


if __name__ == "__main__":
    dense = DenseTensor([2, 5, 2])
    ref = np.zeros([2, 5, 2])
    print(dense[0, slice(None, 4, 2)][slice(0, 3, 3)].to_nparray())
    print(ref[0, slice(None, 4, 2)][slice(0, 3, 3)])

