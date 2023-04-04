import numpy as np


IDEAL_BLOCK_SIZE = 24


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
        item = tuple(item)
        ndim = len(self.shape)

        if len(item) == ndim and all(isinstance(i, int) for i in item):
            coord = [i if i >= 0 else self.shape[x] + i for x, i in enumerate(item)]
            offset = sum(i * stride for i, stride in zip(coord, self.strides))
            return self.get_offset(offset)

        bounds = []
        shape = []

        for x, bound in enumerate(item):
            dim = self.shape[x]

            if isinstance(bound, slice):
                start = 0 if bound.start is None else bound.start if bound.start > 0 else dim + bound.start
                stop = dim if bound.stop is None else bound.stop if bound.stop > 0 else dim + bound.stop
                step = 1 if bound.step is None else bound.step

                assert 0 <= start <= dim
                assert 0 <= stop <= dim
                assert 0 < step <= dim

                bounds.append(slice(start, stop, step))
                shape.append((stop - start) // step)
            elif isinstance(bound, int):
                i = bound if bound >= 0 else dim + bound
                assert 0 <= i <= dim
                bounds.append(i)
            else:
                raise ValueError(f"invalid bound {bound} for ais {x}")

        for dim in self.shape[len(item):]:
            bounds.append(slice(0, dim, 1))
            shape.append(dim)

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
        buffer = [op(ln, rn) for ln, rn in zip(this, that)]  # TODO: is there a way to avoid this allocation?
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

        return BlockView(self, shape, strides)

    def get_offset(self, i):
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
        ndim = len(self.shape)

        if permutation is None:
            permutation = list(reversed(range(ndim)))

        assert len(permutation) == ndim
        assert all(0 <= x < ndim for x in permutation)
        assert all(x in permutation for x in range(ndim))

        shape = tuple(self.shape[x] for x in permutation)
        strides = [self.strides[x] for x in permutation]
        return BlockView(self, shape, strides)


class BlockSlice(Block):
    def __init__(self, source, shape, bounds):
        self.bounds = bounds
        self.shape = tuple(shape)
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

        coord = tuple((i // stride) % dim for dim, stride in zip(self.shape, self.strides))

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
        self.source = source
        self.shape = tuple(shape)
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
        coord = ((i // stride) % dim for dim, stride in zip(self.shape, self.strides))
        i = sum(i * stride for i, stride in zip(coord, self.source_strides))
        return self.source.get_offset(i)


class Tensor(object):
    def __init__(self, shape, data=None):
        assert all(isinstance(dim, int) and dim > 0 for dim in shape), f"invalid shape: {shape}"

        size = np.product(shape)
        assert size

        if size < (2 * IDEAL_BLOCK_SIZE):
            num_blocks = 1
        elif len(shape) == 1 and size % IDEAL_BLOCK_SIZE == 0:
            num_blocks = size // IDEAL_BLOCK_SIZE
        elif len(shape) == 1 or shape[-2] * shape[-1] > (IDEAL_BLOCK_SIZE * 2):
            num_blocks = (size // IDEAL_BLOCK_SIZE)
            num_blocks += 1 if size % IDEAL_BLOCK_SIZE else 0
        else:
            num_blocks = IDEAL_BLOCK_SIZE // (shape[-2] * shape[-1])
            num_blocks += 1 if size % IDEAL_BLOCK_SIZE else 0

        if num_blocks == 1:
            self.blocks = [Buffer(size)]
        else:
            self.blocks = [Buffer(size // num_blocks) for _ in range(num_blocks - 1)]
            if size % num_blocks:
                self.blocks.append(Buffer(size % (size // num_blocks)))

        assert self.blocks

        block_size = len(self.blocks[0])

        if data is not None:
            for offset, n in enumerate(data):
                assert isinstance(n, (complex, float, int)), f"not a number: {n}"
                self.blocks[offset // block_size][offset % block_size] = n

            if offset < size - 1:
                raise ValueError(f"only {offset} elements were provided for a Tensor of size {size}")

        assert size == sum(len(block) for block in self.blocks)
        assert all(len(block) == block_size for block in self.blocks[:-1])
        assert len(self.blocks[-1]) <= block_size

        self.shape = tuple(shape)
        self.strides = strides_for(self.shape)

    def __add__(self, other):
        this, that = broadcast(self, other)
        assert this.shape == that.shape
        return Tensor(this.shape, (lb + rb for lb, rb in zip(this, that)))

    def __matmul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        this, that = broadcast(self, other)
        assert this.shape == that.shape
        return Tensor(this.shape, (lb * rb for lb, rb in zip(this, that)))

    def __eq__(self, other):
        this, that = broadcast(self, other)
        assert this.shape == that.shape
        return Tensor(this.shape, (lb == rb for lb, rb in zip(this, that)))

    def __sub__(self, other):
        this, that = broadcast(self, other)
        assert this.shape == that.shape
        return Tensor(this.shape, (lb - rb for lb, rb in zip(this, that)))

    def __truediv__(self, other):
        this, that = broadcast(self, other)
        assert this.shape == that.shape
        return Tensor(this.shape, (lb / rb for lb, rb in zip(this, that)))

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        for block in self.blocks:
            for n in block:
                yield n

    def __len__(self):
        return np.product(self.shape)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def broadcast(self, shape):
        if self.shape == shape:
            return self

        for dim, bdim in zip(self.shape, shape[-len(self.shape):]):
            if dim == bdim or dim == 1 or bdim == 1:
                pass
            else:
                raise ValueError(f"cannot broadcast dimension {dim} into {bdim}")

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
