import numpy as np

from schema import broadcast, check_permutation, slice_bounds, strides_for


IDEAL_BLOCK_SIZE = 24


class Buffer(object):
    def __init__(self, size, data=None):
        size = int(size)
        assert size

        if data is None:
            self._data = [0] * size
        else:
            self._data = list(data)
            assert size == len(self._data), f"{len(self._data)} elements were provided for a buffer of size {size}"
            assert all(isinstance(n, (complex, float, int)) for n in self._data)

    def __add__(self, other):
        assert len(self) == len(other)
        return Buffer(len(self), [ln + rn for ln, rn in zip(self, other)])

    def __eq__(self, other):
        assert len(self) == len(other)
        return Buffer(len(self), [ln == rn for ln, rn in zip(self, other)])

    def __mod__(self, other):
        assert len(self) == len(other)
        return Buffer(len(self), [ln % rn for ln, rn in zip(self, other)])

    def __mul__(self, other):
        assert len(self) == len(other)
        return Buffer(len(self), [ln * rn for ln, rn in zip(self, other)])

    def __sub__(self, other):
        assert len(self) == len(other)
        return Buffer(len(self), [ln - rn for ln, rn in zip(self, other)])

    def __truediv__(self, other):
        assert len(self) == len(other)
        return Buffer(len(self), [ln / rn for ln, rn in zip(self, other)])

    def __getitem__(self, item):
        if isinstance(item, slice):
            data = self._data[item]
            size = len(data)
            return Buffer(size, data)
        else:
            return self._data[item]

    def __repr__(self):
        return str(list(self))

    def __setitem__(self, key, value):
        if not isinstance(value, (complex, float, int)):
            value = list(value)
            assert len(self._data[key]) == len(value)

        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def reduce_sum(self):
        return sum(iter(self))


class Block(object):
    @staticmethod
    def concatenate(blocks, axis=0):
        assert blocks

        assert all(block.shape[:axis] == tuple(blocks[0].shape[:axis]) for block in blocks)
        assert all(block.shape[axis + 1:] == tuple(blocks[0].shape[axis + 1:]) for block in blocks)

        ndim = len(blocks[0].shape)
        shape = list(blocks[0].shape)
        shape[axis] = sum(block.shape[axis] for block in blocks)
        concatenated = Block(shape)

        offset = 0
        selector = [slice(None) for _ in range(ndim)]
        for i, block in enumerate(blocks):
            selector[axis] = slice(offset, offset + block.shape[axis])
            concatenated[selector] = block
            offset += block.shape[axis]

        return concatenated

    def __init__(self, shape, data=None):
        assert shape

        size = np.product(shape)
        self.buffer = Buffer(size, data)
        self.shape = tuple(int(dim) for dim in shape)
        self.strides = strides_for(shape)

    def __add__(self, other):
        return self._broadcast_op(other, lambda l, r: l + r)

    def __getitem__(self, item):
        item = tuple(item)
        ndim = len(self.shape)

        assert len(item) <= ndim, f"invalid bounds for {self}: {item}"
        assert not any(isinstance(i, np.int64) for i in item)

        if len(item) == ndim and all(isinstance(i, int) for i in item):
            coord = [i if i >= 0 else self.shape[x] + i for x, i in enumerate(item)]
            offset = sum(i * stride for i, stride in zip(coord, self.strides))
            return self.get_offset(offset)

        bounds, shape = slice_bounds(self.shape, item)
        return BlockSlice(self, shape, bounds)

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def __matmul__(self, other):
        ndim = len(self.shape)

        assert ndim >= 2
        assert len(other.shape) == ndim
        assert self.shape[:-2] == other.shape[:-2]

        x, y = self.shape[-2:]
        assert other.shape[-2] == y
        z = other.shape[-1]

        matrix_size = x * z
        num_matrices = len(self) // (x * y)
        assert num_matrices == len(other) // (y * z)

        buffer = Buffer(num_matrices * matrix_size)
        for m in range(num_matrices):
            offset = m * x * z

            for o in range(matrix_size):
                i = o // z
                k = o % z
                buffer[offset + o] = sum(self.get_offset((m * x * y) + (i * x) + (i % x) + j) * other.get_offset((m * y * z) + (j * z) + k) for j in range(y))

        return Block(list(self.shape[:-2]) + [x, z], buffer)

        if ndim == 2:
            shape = (x, z)
            result = Block(shape)
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        result[i, j] = self[i, j] * other[j, k]

            return result
        else:
            raise NotImplementedError

    def __mod__(self, other):
        return self._broadcast_op(other, lambda l, r: l % r)

    def __mul__(self, other):
        return self._broadcast_op(other, lambda l, r: l * r)

    def __setitem__(self, key, value):
        key = tuple(key)
        ndim = len(self.shape)

        if len(key) == ndim and all(isinstance(i, int) for i in key):
            coord = [i if i >= 0 else self.shape[x] + i for x, i in enumerate(key)]
            offset = sum(i * stride for i, stride in zip(coord, self.strides))
            self.buffer[offset] = value
            return

        bounds, shape = slice_bounds(self.shape, key)
        value = value.broadcast(shape)

        for i, n in enumerate(value):
            source_coord = tuple((i // stride) % dim for dim, stride in zip(value.shape, value.strides))

            x = 0
            coord = []
            for bound in bounds:
                if isinstance(bound, int):
                    coord.append(bound)
                else:
                    coord.append(bound.start + (source_coord[x] * bound.step))
                    x += 1

            self[coord] = n

    def __repr__(self):
        return f"(block with shape {self.shape})"

    def __sub__(self, other):
        return self._broadcast_op(other, lambda l, r: l - r)

    def __truediv__(self, other):
        return self._broadcast_op(other, lambda l, r: l // r)

    def _broadcast_op(self, other, op):
        this, that = broadcast(self, other)
        assert this.shape == that.shape
        buffer = [op(ln, rn) for ln, rn in zip(this, that)]
        return Block(this.shape, buffer)

    def broadcast(self, shape):
        assert shape
        assert all(dim > 0 for dim in shape), f"invalid shape for broadcast: {shape}"

        if shape == self.shape:
            return self

        offset = len(shape) - len(self.shape)

        for (ld, rd) in zip(self.shape, shape[offset:]):
            if ld == rd or ld == 1:
                pass
            else:
                raise ValueError(f"cannot broadcast dimensions {ld} and {rd}")

        strides = [0] * len(shape)
        for (x, stride) in enumerate(self.strides):
            strides[offset + x] = stride

        return BlockView(self, shape, strides)

    def get_offset(self, i):
        assert i < np.product(self.shape), f"offset {i} is out of bounds for a Block with shape {self.shape}"
        return self.buffer[i]

    def reduce_sum(self, axes=None):
        if axes is None:
            return sum(self)

        axes = sorted([axes] if isinstance(axes, int) else [int(x) for x in axes])
        shape = [dim for x, dim in enumerate(self.shape) if x not in axes]
        strides = strides_for(shape)

        buffer = Buffer(np.product(shape))

        for i in range(len(buffer)):
            coord = [(i // stride) % dim for dim, stride in zip(shape, strides)]
            for x in axes:
                coord.insert(x, slice(None))

            buffer[i] = sum(self[coord])

        return Block(shape, buffer)

    def transpose(self, permutation=None):
        permutation = check_permutation(self.shape, permutation)

        if all(i == x for i, x in enumerate(permutation)):
            return self

        shape = tuple(self.shape[x] for x in permutation)
        strides = [self.strides[x] for x in permutation]
        return BlockView(self, shape, strides)


class BlockSlice(Block):
    def __init__(self, source, shape, bounds):
        assert shape

        self.bounds = bounds
        self.shape = tuple(int(dim) for dim in shape)
        self.source = source
        self.strides = strides_for(shape)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_offset(i)

    def __len__(self):
        return np.product(self.shape)

    def __setitem__(self, key, value):
        raise NotImplementedError("cannot write to a BlockSlice")

    def get_offset(self, i):
        assert i < len(self)

        coord = tuple((i // stride) % dim if stride else 0 for dim, stride in zip(self.shape, self.strides))

        x = 0
        source_coord = []
        for bound in self.bounds:
            if isinstance(bound, int):
                source_coord.append(bound)
            else:
                source_coord.append(bound.start + (coord[x] * bound.step))
                x += 1

        return self.source[source_coord]


class BlockView(Block):
    def __init__(self, source, shape, strides):
        assert shape

        self.source = source
        self.shape = tuple(int(dim) for dim in shape)
        self.source_strides = strides
        self.strides = strides_for(shape)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_offset(i)

    def __len__(self):
        return int(np.product(self.shape))

    def __setitem__(self, key, value):
        raise NotImplementedError("cannot write to a BlockView")

    def get_offset(self, i):
        assert i < len(self)
        coord = [(i // stride) % dim if stride else 0 for dim, stride in zip(self.shape, self.strides)]
        i = sum(i * stride for i, stride in zip(coord, self.source_strides))
        return self.source.get_offset(i)
