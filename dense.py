import itertools
import math
import numpy as np
import sys

from base import Broadcast, Expansion, Rebase, Tensor, TensorSlice
from base import affected, product, validate_match


BLOCK_SIZE = 1000
BLOCK_OF_COORDS_LEN = BLOCK_SIZE // sys.getsizeof(np.uint64())


class BlockTensorView(Tensor):
    def __init__(self, shape, dtype, per_block = None):
        Tensor.__init__(self, shape, dtype)

        if per_block is None:
            per_block = BLOCK_SIZE // sys.getsizeof(dtype())
        self._per_block = per_block

    def __eq__(self, other):
        if not hasattr(other, "shape") or other.shape == tuple():
            equal_blocks = []
            for block in self.blocks():
                equal_blocks.append(block == other)
            return BlockTensor(self.shape, np.bool, equal_blocks, self._per_block)
        elif not isinstance(other, BlockTensorView):
            return Tensor.__eq__(other, self)

        shape = [max(l, r) for l, r in zip(self.shape, other.shape)]
        this = self.broadcast(shape)
        that = other.broadcast(shape)

        these_blocks = this.blocks()
        those_blocks = that.blocks()
        equal_blocks = []
        while True:
            try:
                this_block = next(these_blocks)
                that_block = next(those_blocks)
                assert len(this_block) == len(that_block)

                equal_block = this_block == that_block
                assert len(equal_block) == len(this_block)
                equal_blocks.append(equal_block)
            except StopIteration:
                break

        return BlockTensor(this.shape, np.bool, equal_blocks, self._per_block)

    def __mul__(self, other):
        if not hasattr(other, "shape") or other.shape == tuple():
            mul_blocks = []
            for block in self.blocks():
                mul_blocks.append(block * other)
            return BlockTensor(self.shape, self.dtype, mul_blocks, self._per_block)
        elif not isinstance(other, BlockTensorView):
            return other * self

        shape = [max(l, r) for l, r in zip(self.shape, other.shape)]
        this = self.broadcast(shape)
        that = other.broadcast(shape)

        these_blocks = this.blocks()
        those_blocks = that.blocks()
        mul_blocks = []
        while True:
            try:
                this_block = next(these_blocks)
                that_block = next(those_blocks)
                assert len(this_block) == len(that_block)

                mul_block = this_block * that_block
                assert len(mul_block) == len(this_block)
                mul_blocks.append(mul_block)
            except StopIteration:
                break

        return BlockTensor(this.shape, this.dtype, mul_blocks, this._per_block)

    def blocks(self):
        raise NotImplementedError

    def broadcast(self, shape):
        if shape == self.shape:
            return self

        return BlockTensorBroadcast(self, shape)

    def expand_dims(self, axis):
        return BlockTensorExpansion(self, axis)

    def product(self, axis = None):
        if axis is None or (axis == 0 and self.ndim == 1):
            product = 1
            for block in self.blocks():
                product *= np.product(block)
            return product

        assert axis < self.ndim
        shape = list(self.shape)
        del shape[axis]
        product = BlockTensor(shape, self.dtype)

        if axis == 0:
            for coord in itertools.product(*[range(dim) for dim in shape]):
                source_coord = (slice(None),) + coord
                product[coord] = self[source_coord].product()
        else:
            prefix_range = [range(self.shape[x]) for x in range(axis)]
            for prefix in itertools.product(*prefix_range):
                product[prefix] = self[prefix].product(0)

        return product

    def sum(self, axis = None):
        if axis is None or (axis == 0 and self.ndim == 1):
            summed = 0
            for block in self.blocks():
                summed += np.sum(block)
            return summed

        assert axis < self.ndim
        shape = list(self.shape)
        del shape[axis]
        summed = BlockTensor(shape, self.dtype)

        if axis == 0:
            for coord in itertools.product(*[range(dim) for dim in shape]):
                source_coord = (slice(None),) + coord
                summed[coord] = self[source_coord].sum()
        else:
            prefix_range = [range(self.shape[x]) for x in range(axis)]
            for prefix in itertools.product(*prefix_range):
                summed[prefix] = self[prefix].sum(0)

        return summed


class BlockTensor(BlockTensorView):
    def __init__(self, shape, dtype=np.int32, blocks=None, per_block=None):
        BlockTensorView.__init__(self, shape, dtype, per_block)

        if blocks:
            expected_block_len = self.size if len(blocks) == 1 else per_block
            assert len(blocks[0]) == expected_block_len
            self._blocks = blocks
        else:
            self._blocks = [
                np.zeros([self._per_block], dtype)
                for _ in range(self.size // self._per_block)]

            if self.size % self._per_block > 0:
                self._blocks.append(np.zeros([self.size % self._per_block], dtype))

        self._coord_index = np.array([product(shape[axis + 1:]) for axis in range(self.ndim)])

    def __getitem__(self, match):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            index = sum(np.array(match) * self._coord_index)
            return self._blocks[index // self._per_block][index % self._per_block]
        else:
            return BlockTensorSlice(self, match)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            index = sum(np.array(match) * self._coord_index)
            self._blocks[index // self._per_block][index % self._per_block] = self.dtype(value)
        elif isinstance(value, Tensor):
            Tensor.__setitem__(self, match, value)
        else:
            affected_range = itertools.product(*affected(match, self.shape))
            value = self.dtype(value)
            for coords in chunk_iter(affected_range, BLOCK_OF_COORDS_LEN):
                coords = [list(coord) for coord in coords]
                block = np.array(coords) * self._coord_index
                while block.ndim > 1:
                    block = block.sum(1)

                indices = block // self._per_block
                offsets = block % self._per_block
                for i, o in zip(indices, offsets):
                    self._blocks[i][o] = value

    def blocks(self):
        yield from (block for block in self._blocks)


class BlockTensorDerived(BlockTensorView):
    def __init__(self, source, shape):
        BlockTensorView.__init__(self, shape, source.dtype)

    def blocks(self):
        coord_range = itertools.product(*[range(dim) for dim in self.shape])
        for coords in chunk_iter(coord_range, self._source._per_block):
            yield np.array([self[coord] for coord in coords])


class BlockTensorBroadcast(BlockTensorDerived, Broadcast):
    def __init__(self, source, shape):
        Broadcast.__init__(self, source, shape)
        BlockTensorDerived.__init__(self, source, shape)


class BlockTensorExpansion(BlockTensorDerived, Expansion):
    def __init__(self, source, axis):
        Expansion.__init__(self, source, axis)
        BlockTensorDerived.__init__(self, source, self.shape)


class BlockTensorSlice(BlockTensorDerived, TensorSlice):
    def __init__(self, source, match):
        TensorSlice.__init__(self, source, match)
        BlockTensorDerived.__init__(self, source, self.shape)


def chunk_iter(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk

