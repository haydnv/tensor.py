import numpy as np


class Buffer(object):
    def __init__(self, size, data=None):
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

    def __setitem__(self, key, value):
        value = list(value)
        assert len(self._data[key]) == len(value)
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class Coords(object):
    @classmethod
    def from_offsets(cls, shape, offsets):
        ndim = len(shape)
        size = len(offsets)

        offsets = Block((size, 1), offsets)
        strides = Block((ndim,), (int(np.product(shape[i + 1:])) for i in range(ndim)))
        coords = (offsets / strides) % Block((ndim,), shape)

        return cls(shape, size, coords)

    def __init__(self, shape, size, coords=None):
        self.buffer = Buffer(len(shape) * size, coords)
        self.shape = shape

    def __len__(self):
        return len(self.buffer) // len(self.shape)

    def to_offsets(self):
        ndim = len(self.shape)

        strides = Block((ndim,), (int(np.product(self.shape[i + 1:])) for i in range(ndim)))
        coords = Block((len(self), ndim), self.buffer)
        return (coords * strides).reduce_sum(0)


class Block(object):
    def __init__(self, shape, data=None):
        size = np.product(shape)
        self.buffer = Buffer(size, data)
        self.shape = shape

    def __add__(self, other):
        assert self.shape == other.shape
        return Block(self.shape, self.buffer + other.buffer)

    def __matmul__(self, other):
        assert self.shape == other.shape
        return Block(self.shape, self.buffer * other.buffer)

    def __mul__(self, other):
        assert self.shape == other.shape
        return Block(self.shape, self.buffer * other.buffer)

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        return len(self.buffer)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __sub__(self, other):
        assert self.shape == other.shape
        return Block(self.shape, self.buffer - other.buffer)

    def __truediv__(self, other):
        assert self.shape == other.shape
        return Block(self.shape, self.buffer / other.buffer)

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

        self.shape = shape

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

    def transpose(self, permutation=None):
        raise NotImplementedError


def broadcast(left, right):
    raise NotImplementedError
