import numpy as np


class Buffer(object):
    def __init__(self, size, data=None):
        size = int(size)

        if data is None:
            self._data = [0] * size
        else:
            self._data = list(data)
            assert size == len(self._data)
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
    def __init__(self, shape, data=None):
        size = np.product(shape)
        self.buffer = Buffer(size, data)
        self.shape = tuple(shape)
        self.strides = strides_for(shape)

    def __add__(self, other):
        return self._broadcast_op(other, lambda l, r: l + r)

    def __getitem__(self, item):
        raise NotImplementedError(f"get item {item} from block with shape {self.shape}")

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def __matmul__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        return self._broadcast_op(other, lambda l, r: l % r)

    def __mul__(self, other):
        return self._broadcast_op(other, lambda l, r: l * r)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        return f"(block with shape {self.shape})"

    def __sub__(self, other):
        return self._broadcast_op(other, lambda l, r: l - r)

    def __truediv__(self, other):
        return self._broadcast_op(other, lambda l, r: l // r)

    def _broadcast_op(self, other, op):
        this, that = broadcast(self, other)
        assert this.shape == that.shape
        buffer = [op(ln, rn) for ln, rn in zip(iter(self), iter(that))]
        return Block(this.shape, buffer)

    def broadcast(self, shape):
        if shape == self.shape:
            return self

        offset = len(shape) - len(self.shape)

        for (ld, rd) in zip(self.shape, shape[offset:]):
            if ld == rd or ld == 1 or rd == 1:
                pass
            else:
                raise ValueError(f"cannot broadcast dimensions {ld} and {rd}")

        strides = [0] * len(shape)
        for (x, stride) in enumerate(self.strides):
            strides[offset + x] = stride

        return BlockView(shape, self, strides)

    def get_offset(self, i):
        return self.buffer[i]

    def reduce_sum(self, axes=None):
        if axes is None:
            return self.buffer.reduce_sum()

        axes = sorted([axes] if isinstance(axes, int) else [int(x) for x in axes])
        shape = list(self.shape)
        source = self.buffer

        while axes:
            axis = axes.pop()
            dim = shape.pop(axis)
            size = len(source) // dim
            stride = int(np.product(shape[axis:]))
            length = stride * dim
            buffer = Buffer(size)  # TODO: is there a way to avoid this allocation?

            for i in range(size):
                x = (i // stride) * length
                x_i = i % stride
                buffer[i] = source[x + x_i:x + x_i + length:stride].reduce_sum()

            source = buffer

        return Block(shape, source)

    def transpose(self, permutation=None):
        ndim = len(self.shape)

        if permutation is None:
            permutation = list(reversed(range(ndim)))

        assert len(permutation) == ndim
        assert all(0 <= x < ndim for x in permutation)
        assert all(x in permutation for x in range(ndim))

        shape = tuple(self.shape[x] for x in permutation)
        strides = [self.strides[x] for x in permutation]
        return BlockView(shape, self, strides)


class BlockView(Block):
    def __init__(self, shape, source, strides):
        self.shape = shape
        self.source = source
        self.source_strides = strides
        self.strides = strides_for(shape)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_offset(i)

    def __len__(self):
        return int(np.product(self.shape))

    def get_offset(self, i):
        coord = ((i // stride) % dim for dim, stride in zip(self.shape, self.strides))
        i = sum(i * stride for i, stride in zip(coord, self.source_strides))
        return self.source.get_offset(i)

    def reduce_sum(self, axes=None):
        raise NotImplementedError


class Tensor(object):
    def __init__(self, shape, blocks=None):
        assert all(isinstance(dim, int) and dim > 0 for dim in shape)

        size = np.product(shape)

        if blocks is None:
            self.blocks = [Buffer(size)]
        else:
            self.blocks = [Buffer(len(block), block) for block in blocks]
            assert size == sum(len(block) for block in self.blocks)

        self.shape = tuple(shape)

    def __add__(self, other):
        assert self.shape == other.shape
        assert len(self.blocks) == len(other.blocks)
        return Tensor(self.shape, (lb + rb for lb, rb in zip(self, other)))

    def __matmul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        assert self.shape == other.shape
        assert len(self.blocks) == len(other.blocks)
        return Tensor(self.shape, (lb * rb for lb, rb in zip(self, other)))

    def __eq__(self, other):
        assert self.shape == other.shape
        assert len(self.blocks) == len(other.blocks)
        return Tensor(self.shape, (lb == rb for lb, rb in zip(self, other)))

    def __sub__(self, other):
        assert self.shape == other.shape
        assert len(self.blocks) == len(other.blocks)
        return Tensor(self.shape, (lb - rb for lb, rb in zip(self, other)))

    def __truediv__(self, other):
        assert self.shape == other.shape
        assert len(self.blocks) == len(other.blocks)
        return Tensor(self.shape, (lb - rb for lb, rb in zip(self, other)))

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return np.product(self.shape)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def reduce_sum(self, axes=None):
        if axes is None:
            return sum(block.reduce_sum() for block in self.blocks)

        axes = sorted([axes] if isinstance(axes, int) else axes)
        raise NotImplementedError

    def transpose(self, permutation=None):
        raise NotImplementedError


def broadcast(left, right):
    if len(left.shape) < len(right.shape):
        right, left = broadcast(right, left)
        return left, right

    shape = left.shape
    offset = len(shape) - len(right.shape)
    for x in range(len(right.shape)):
        if shape[offset + x] == right.shape[x]:
            pass
        elif right.shape[x] == 1:
            pass
        elif shape[offset + x] == 1:
            shape[offset + x] = right.shape[x]
        else:
            raise ValueError(f"cannot broadcast dimensions {shape[offset + x]} and {right.shape[x]}")

    return left.broadcast(shape), right.broadcast(shape)


def strides_for(shape):
    return [int(np.product(shape[x + 1:])) for x in range(len(shape))]
