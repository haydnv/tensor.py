import numpy as np


IDEAL_BLOCK_SIZE = 24


class Buffer(object):
    def __init__(self, size, data=None):
        size = int(size)

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
        assert shape

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


class Tensor(object):
    def __init__(self, shape, data=None):
        assert all(isinstance(dim, int) and dim > 0 for dim in shape), f"invalid shape: {shape}"

        shape = list(shape)
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

        block_size = size // num_blocks

        block_axis = 0
        while np.product(shape[block_axis:]) < block_size:
            block_axis += 1

        if num_blocks == 1:
            buffers = [Buffer(block_size)]
        else:
            buffers = [Buffer(block_size) for _ in range(num_blocks - 1)]

            if size % num_blocks:
                buffers.append(Buffer(size % block_size))

        block_shape = shape[block_axis:]

        if data is not None:
            for offset, n in enumerate(data):
                assert isinstance(n, (complex, float, int)), f"not a number: {n}"
                buffers[offset // block_size][offset % block_size] = n

            if offset + 1 != size:
                raise ValueError(f"{offset + 1} elements were provided for a Tensor of size {size}")

        if size % block_size:
            last_block_shape = block_shape
            last_block_shape[0] = int(block_shape[0] // np.product(block_shape[1:]))

            self.blocks = [Block(block_shape, buffer) for buffer in buffers[:-1]]
            self.blocks.append(Block(last_block_shape, buffers[-1]))
        else:
            self.blocks = [Block(block_shape, buffer) for buffer in buffers]

        assert self.blocks
        assert size == sum(len(block) for block in self.blocks)
        assert all(len(block) == block_size for block in self.blocks[:-1])
        assert len(self.blocks[-1]) <= block_size

        map_shape = shape[:block_axis] + [int(np.product(shape[block_axis:]) // block_size)]
        self.block_map = Block(map_shape, range(len(self.blocks)))
        self.shape = tuple(int(dim) for dim in shape)

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
        item = tuple(item)
        ndim = len(self.shape)
        block_size = len(self.blocks[0])

        if len(item) == ndim and all(isinstance(i, int) for i in item):
            coord = [i if i >= 0 else self.shape[x] + i for x, i in enumerate(item)]
            offset = sum(i * stride for i, stride in zip(coord, strides_for(self.shape)))
            return self.blocks[offset // block_size][offset % block_size]

        bounds, shape = slice_bounds(self.shape, item)
        return TensorSlice(self, shape, bounds)

    def __iter__(self):
        for i in self.block_map:
            for n in self.get_block(i):
                yield n

    def __len__(self):
        return np.product(self.shape)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def broadcast(self, shape):
        if self.shape == shape:
            return self

        # characterize the source tensor (this tensor)
        block_size = len(self) // len(self.block_map)
        block_axis = 0
        while np.product(shape[block_axis:]) < block_size:
            block_axis += 1

        # characterize the output tensor (the broadcasted view of this tensor)
        shape = list(shape)
        offset = len(shape) - len(self.shape)
        block_shape = shape[offset + block_axis:]
        map_shape = shape[:offset + block_axis]

        block_map = self.block_map.broadcast(map_shape) if map_shape else self.block_map
        blocks = [self.get_block(i).broadcast(block_shape) for i in block_map]

        return TensorView(blocks, block_map, shape)

    def get_block(self, i):
        return self.blocks[int(i)]

    def reduce_sum(self, axes=None):
        if axes is None:
            return sum(block.reduce_sum() for block in self.blocks)

        axes = sorted([axes] if isinstance(axes, int) else axes)
        raise NotImplementedError

    def reshape(self, shape):
        if shape == self.shape:
            return self
        elif np.product(shape) == np.product(self.shape):
            return Tensor(shape, self)
        else:
            raise ValueError(f"cannot reshape from {self.shape} into {shape}")

    def transpose(self, permutation=None):
        permutation = check_permutation(self.shape, permutation)
        shape = tuple(self.shape[x] for x in permutation)
        raise NotImplementedError("Tensor.transpose")


class TensorSlice(Tensor):
    def __init__(self, source, shape, bounds):
        self.bounds = bounds
        self.shape = shape
        self.source = source

        self.block_map = "TODO"

    def get_block(self, i):
        raise NotImplementedError


class TensorView(Tensor):
    def __init__(self, blocks, block_map, shape):
        self.blocks = blocks
        self.block_map = block_map
        self.shape = tuple(int(dim) for dim in shape)

    def get_block(self, i):
        i = int(i)
        strides = strides_for(self.block_map.shape)
        coord = [(i // stride) + (i % dim) if stride else i % dim for stride, dim in zip(strides, self.block_map.shape)]
        i = self.block_map[coord]
        return self.blocks[i]


def broadcast(left, right):
    if len(left.shape) < len(right.shape):
        right, left = broadcast(right, left)
        return left, right

    shape = list(left.shape)
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


def check_permutation(shape, permutation):
    ndim = len(shape)

    if permutation is None:
        permutation = list(reversed(range(ndim)))
    else:
        permutation = [ndim + x if x < 0 else x for x in permutation]

    assert len(permutation) == ndim
    assert all(0 <= x < ndim for x in permutation)
    assert all(x in permutation for x in range(ndim))

    return permutation


def slice_bounds(source_shape, key):
    bounds = []
    shape = []

    for x, bound in enumerate(key):
        dim = source_shape[x]

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
            raise ValueError(f"invalid bound {bound} (type {type(bound)}) for axis {x}")

    for dim in source_shape[len(key):]:
        bounds.append(slice(0, dim, 1))
        shape.append(dim)

    return bounds, shape


def strides_for(shape):
    return [0 if dim == 1 else int(np.product(shape[x + 1:])) for x, dim in enumerate(shape)]
