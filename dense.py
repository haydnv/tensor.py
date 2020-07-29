import itertools
import math
import numpy as np
import sys

import transform

from base import Tensor, affected, product, validate_match

PER_BLOCK = 10


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
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.size = product(shape)

    def broadcast(self, shape):
        return BlockListBroadcast(self, shape)


class BlockListBase(BlockList):
    @staticmethod
    def constant(shape, dtype, value):
        assert isinstance(value, dtype)

        size = product(shape)

        blocks = [np.ones([PER_BLOCK], dtype) * value for _ in range(size // PER_BLOCK)]
        if size % PER_BLOCK > 0:
            blocks.append(np.ones([size % PER_BLOCK], dtype) * value)

        return BlockListBase(shape, dtype, blocks)

    def __init__(self, shape, dtype, blocks=None):
        BlockList.__init__(self, shape, dtype)

        if blocks is None:
            blocks = [np.zeros([PER_BLOCK], dtype) for _ in range(self.size // PER_BLOCK)]
            if self.size % PER_BLOCK > 0:
                blocks.append(np.zeros([self.size % PER_BLOCK], dtype))
        else:
            assert len(blocks) == math.ceil(self.size / PER_BLOCK)
            for block in blocks[:-1]:
                assert block.shape == (PER_BLOCK,)

            if self.size % PER_BLOCK:
                assert blocks[-1].shape == (self.size % PER_BLOCK,)

        self._blocks = blocks
        self._coord_index = np.array(
            [product(shape[axis + 1:]) for axis in range(self.ndim)])

    def __getitem__(self, match):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            index = sum(np.array(match) * self._coord_index)
            return self._blocks[index // PER_BLOCK][index % PER_BLOCK]
        else:
            return BlockListSlice(self, match)

    def __iter__(self):
        return iter(self._blocks)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            index = sum(np.array(match) * self._coord_index)
            self._blocks[index // PER_BLOCK][index % PER_BLOCK] = self.dtype(value)
        else:
            update_shape = shape_of(self.shape, match)

            if isinstance(value, Tensor):
                if not isinstance(value, DenseTensor):
                    value = DenseTensor.from_sparse(value)

                if value.shape != update_shape:
                    value = value.broadcast(update_shape)
                value = list(iter(value))
            else:
                value = list(iter(value for _ in range(product(update_shape))))

            assert len(value) == product(update_shape)

            affected_coords = itertools.product(*affected(match, self.shape))
            for coords in chunk_iter(affected_coords, PER_BLOCK):
                indices = np.sum(coords * self._coord_index, 1)
                block_ids = indices // PER_BLOCK
                offsets = indices % PER_BLOCK
                for block_id in np.unique(block_ids):
                    num_values = np.sum(block_ids == block_id)
                    block_offsets = offsets[:num_values]
                    block_values = value[:num_values]
                    self._blocks[block_id][block_offsets] = np.array(block_values)

                    offsets = offsets[num_values:]
                    value = value[num_values:]


class BlockListRebase(BlockList):
    def __init__(self, source, rebase):
        BlockList.__init__(self, rebase.shape, source.dtype)
        self._source = source
        self._rebase = rebase

    def __getitem__(self, match):
        match = validate_match(match, self.shape)
        return self._source[self._rebase.invert_coord(match)]

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)
        self._source[self._rebase.invert_coord(match)] = value


class BlockListBroadcast(BlockListRebase):
    def __init__(self, source, shape):
        rebase = transform.Broadcast(source.shape, shape)
        BlockListRebase.__init__(self, source, rebase)

    def __iter__(self):
        coord_range = itertools.product(*[range(dim) for dim in self.shape])
        for coords in chunk_iter(coord_range, PER_BLOCK):
            yield np.array([self[coord] for coord in coords])


class BlockListSlice(BlockListRebase):
    def __init__(self, source, match):
        rebase = transform.Slice(source.shape, match)
        BlockListRebase.__init__(self, source, rebase)

    def __iter__(self):
        match = affected(self._rebase.match, self._source.shape)
        for coords in chunk_iter(itertools.product(*match), PER_BLOCK):
            yield np.array([self._source[coord] for coord in coords])


class BlockListSparse(BlockList):
    def __init__(self, source):
        self.dtype = source.dtype
        self.ndim = len(source.shape)
        self.shape = source.shape
        self.size = product(self.shape)

        self._coord_index = np.array(
            [product(source.shape[axis + 1:]) for axis in range(self.ndim)])
        self._source = source

    def __getitem__(self, match):
        match = validate_match(match, self.shape)
        if len(match) == self.ndim and all(isinstance(x, int) for x in match):
            return self._source[match]
        else:
            return BlockListSparse(self._source[match])

    def __iter__(self):
        shape = np.array(self.shape)
        for offset in range(PER_BLOCK, self.size, PER_BLOCK):
            # TODO: query source.filled_in instead
            block = []
            offsets = np.expand_dims(np.arange(offset - PER_BLOCK, offset), 1)
            coords = np.reshape((offsets // self._coord_index) % shape, [PER_BLOCK, self.ndim])
            for coord in coords:
                coord = tuple(int(c) for c in coord)
                block.append(self._source[coord])

            yield np.array(block)

        trailing_len = self.size % PER_BLOCK
        trailing_len = trailing_len if trailing_len else PER_BLOCK
        block = []
        offsets = np.expand_dims(np.arange(self.size - trailing_len, self.size), 1)
        coords = np.reshape((offsets // self._coord_index) % shape, [trailing_len, self.ndim])
        for coord in coords:
            coord = tuple(int(c) for c in coord)
            block.append(self._source[coord])

        yield np.array(block)

    def broadcast(self, shape):
        return BlockListSparse(self._source.broadcast(shape))


class DenseTensor(Tensor):
    @staticmethod
    def from_sparse(sparse_tensor):
        block_list = BlockListSparse(sparse_tensor)
        return DenseTensor(sparse_tensor.shape, sparse_tensor.dtype, block_list)

    @staticmethod
    def ones(shape, dtype=np.int32):
        block_list = BlockListBase.constant(shape, dtype, dtype(1))
        return DenseTensor(shape, dtype, block_list)

    def __init__(self, shape, dtype=np.int32, block_list=None):
        Tensor.__init__(self, shape, dtype)

        if block_list is None:
            self._block_list = BlockListBase(shape, dtype)
        else:
            assert isinstance(block_list, BlockList)
            assert block_list.dtype == self.dtype
            assert block_list.shape == self.shape
            self._block_list = block_list

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            blocks = [block == other for block in self._block_list]
            block_list = BlockListBase(self.shape, np.bool, blocks)
            return DenseTensor(self.shape, np.bool, block_list)
        elif not isinstance(other, DenseTensor):
            other = DenseTensor.from_sparse(other)

        if other.shape != self.shape:
            other = other.broadcast(self.shape)
        if self.shape != other.shape:
            return self.broadcast(other.shape) == other

        left = list(iter(self._block_list))
        right = list(iter(other._block_list))
        assert len(left) == len(right)
        blocks = list(zip(left, right))
        for l, r in blocks:
            assert l.shape == r.shape
        blocks = [left == right for left, right in blocks]
        block_list = BlockListBase(self.shape, np.bool, blocks)
        return DenseTensor(self.shape, np.bool, block_list)

    def __getitem__(self, match):
        if not isinstance(match, tuple):
            match = (match,)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            match = validate_match(match, self.shape)
            return self._block_list[match]
        else:
            block_list = self._block_list[match]
            return DenseTensor(block_list.shape, self.dtype, block_list)

    def __iter__(self):
        for block in self._block_list:
            for i in range(block.size):
                yield block[i]

    def __mul__(self, other):
        if not hasattr(other, "shape") or other.shape == tuple():
            blocks = [block * other for block in self._block_list]
            block_list = BlockListBase(self.shape, self.dtype, blocks)
            return DenseTensor(self.shape, self.dtype, block_list)
        elif not isinstance(other, DenseTensor):
            return other * self

        shape = [max(l, r) for l, r in zip(self.shape, other.shape)]
        this = self.broadcast(shape)
        that = other.broadcast(shape)

        these_blocks = iter(this._block_list)
        those_blocks = iter(that._block_list)
        blocks = []
        while True:
            try:
                this_block = next(these_blocks)
                that_block = next(those_blocks)
                assert len(this_block) == len(that_block)

                block = this_block * that_block
                assert len(block) == len(this_block)
                blocks.append(block)
            except StopIteration:
                break

        block_list = BlockListBase(shape, self.dtype, blocks)
        return DenseTensor(shape, self.dtype, block_list)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)
        self._block_list[match] = value

    def all(self):
        for block in self._block_list:
            if not block.all():
                return False

        return True

    def broadcast(self, shape):
        if shape == self.shape:
            return self

        block_list = self._block_list.broadcast(shape)
        return DenseTensor(shape, self.dtype, block_list)

    def product(self, axis = None):
        if axis is None or (axis == 0 and self.ndim == 1):
            multiplied = self.dtype(1)
            for block in self._block_list:
                multiplied *= np.product(block)
            return multiplied

        assert axis < self.ndim
        shape = list(self.shape)
        del shape[axis]
        multiplied = DenseTensor(shape, self.dtype)

        if axis == 0:
            for coord in itertools.product(*[range(dim) for dim in shape]):
                source_coord = (slice(None),) + coord
                multiplied[coord] = self[source_coord].product()
        else:
            prefix_range = [range(dim) for dim in self.shape[:axis]]
            for prefix in itertools.product(*prefix_range):
                multiplied[prefix] = self[prefix].product(0)

        return multiplied

    def sum(self, axis = None):
        if axis is None or (axis == 0 and self.ndim == 1):
            summed = 0
            for block in self._block_list:
                summed += np.sum(block)
            return summed

        assert axis < self.ndim
        shape = list(self.shape)
        del shape[axis]
        summed = DenseTensor(shape, self.dtype)

        if axis == 0:
            for coord in itertools.product(*[range(dim) for dim in shape]):
                source_coord = (slice(None),) + coord
                summed[coord] = self[source_coord].sum()
        else:
            prefix_range = [range(dim) for dim in self.shape[:axis]]
            for prefix in itertools.product(*prefix_range):
                summed[prefix] = self[prefix].sum(0)

        return summed

    def to_nparray(self):
        arr = np.zeros(self.shape, self.dtype)
        for coord in itertools.product(*[range(dim) for dim in self.shape]):
            arr[coord] = self[coord]
        return arr


def chunk_iter(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk

