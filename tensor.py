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


class Coords(object):
    @classmethod
    def from_offsets(cls, shape, offsets):
        ndim = len(shape)
        size = len(offsets)

        offsets = Block((size, 1), offsets)
        strides = Block((ndim, 1), (int(np.product(shape[i + 1:])) for i in range(ndim)))
        coords = (offsets / strides) % Block((ndim, 1), shape)

        return cls(shape, size, coords)

    def __init__(self, shape, size, coords=None):
        self.buffer = Buffer(len(shape) * size, coords)
        self.shape = shape

    def __len__(self):
        return len(self.buffer) // len(self.shape)

    def __repr__(self):
        ndim = len(self.shape)
        return str([self.buffer[i:i + ndim] for i in range(0, len(self.buffer), ndim)])

    def to_offsets(self):
        ndim = len(self.shape)

        strides = Block((ndim,), (int(np.product(self.shape[i + 1:])) for i in range(ndim)))
        coords = Block((len(self), ndim), self.buffer)
        return (coords * strides).reduce_sum(0)


class Block(object):
    def __init__(self, shape, data=None):
        size = np.product(shape)
        self.buffer = Buffer(size, data)
        self.shape = tuple(shape)

    def __add__(self, other):
        return self._broadcast(other, lambda l, r: l + r)

    def __getitem__(self, item):
        raise NotImplementedError(f"get item {item} from block with shape {self.shape}")

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def __matmul__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        return self._broadcast(other, lambda l, r: l % r)

    def __mul__(self, other):
        return self._broadcast(other, lambda l, r: l * r)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        return f"(block with shape {self.shape})"

    def __sub__(self, other):
        return self._broadcast(other, lambda l, r: l // r)

    def __truediv__(self, other):
        return self._broadcast(other, lambda l, r: l // r)

    def _broadcast(self, other, op):
        if self.shape == other.shape:
            shape = self.shape
            buffer = [op(self.buffer[i], other.buffer[i]) for i in range(len(self))]
        else:
            shape = broadcast_into(other.shape, self.shape)
            buffer = [op(self.buffer[i], other.buffer[i % len(other)]) for i in range(len(self))]

        return Block(shape, buffer)

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
            stride = np.product(shape[axis:])
            length = stride * dim
            buffer = Buffer(size)

            for i in range(size):
                x = (i // stride) * length
                x_i = i % stride
                buffer[i] = source[x + x_i:x + x_i + length:stride].reduce_sum()

            source = buffer

        return Block(shape, source)

    def transpose(self, permutation=None):
        if permutation is None:
            permutation = list(reversed(range(len(self.shape))))

        assert len(permutation) == len(self.shape)

        buffer = Buffer(len(self), self.buffer)
        for i, x in enumerate(permutation):
            if i == x:
                pass  # nothing to do
            else:
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


def broadcast_into(small, big):
    if len(small) > len(big):
        raise ValueError(f"cannot broadcast {small} into {big}")

    shape = list(big)

    offset = len(big) - len(small)
    for x in range(len(small)):
        if small[x] == big[x + offset]:
            pass
        elif small[x] == 1:
            pass
        else:
            raise ValueError(f"cannot broadcast dimension {small[x]} into {big[x + offset]}")

    return tuple(shape)
