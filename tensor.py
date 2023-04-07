import numpy as np

from block import Buffer, Block
from schema import broadcast, check_permutation, slice_bounds, strides_for


IDEAL_BLOCK_SIZE = 24


class Tensor(object):
    def __init__(self, shape, data=None):
        assert all(isinstance(dim, int) and dim > 0 for dim in shape), f"invalid shape: {shape}"

        shape = list(shape)
        size = np.product(shape)
        assert size

        if size < (2 * IDEAL_BLOCK_SIZE):
            block_size = size
            num_blocks = 1
        elif len(shape) == 1 and size % IDEAL_BLOCK_SIZE == 0:
            block_size = IDEAL_BLOCK_SIZE
            num_blocks = size // IDEAL_BLOCK_SIZE
        elif len(shape) == 1 or shape[-2] * shape[-1] > (IDEAL_BLOCK_SIZE * 2):
            block_size = IDEAL_BLOCK_SIZE
            num_blocks = (size // IDEAL_BLOCK_SIZE)
            num_blocks += 1 if size % IDEAL_BLOCK_SIZE else 0
        else:
            matrix_size = shape[-2] * shape[-1]
            block_size = IDEAL_BLOCK_SIZE + (matrix_size - (IDEAL_BLOCK_SIZE % matrix_size))
            num_blocks = size // block_size
            num_blocks += 1 if size % block_size else 0

        assert block_size

        if size % block_size:
            buffers = [Buffer(block_size) for _ in range(num_blocks - 1)]
            buffers.append(Buffer(size % block_size))
        else:
            buffers = [Buffer(block_size) for _ in range(num_blocks)]

        if data is not None:
            for offset, n in enumerate(data):
                assert isinstance(n, (complex, float, int)), f"not a number: {n}"
                buffer = buffers[offset // block_size]
                buffer[offset % block_size] = n

        block_axis = block_axis_for(shape, block_size)

        block_shape = shape[block_axis:]
        block_shape[0] = int(np.ceil(block_size / np.product(block_shape[1:])))
        map_shape = shape[:block_axis]
        map_shape.append(int(np.ceil(shape[block_axis] / block_shape[0])))

        assert len(block_shape) + len(map_shape) == len(shape) + 1

        if size % block_size:
            last_block_shape = list(block_shape)
            last_block_shape[0] = len(buffers[-1]) // np.product(block_shape[1:])
            self.blocks = [Block(block_shape, buffer) for buffer in buffers[:-1]]
            self.blocks.append(Block(last_block_shape, buffers[-1]))
        else:
            self.blocks = [Block(block_shape, buffer) for buffer in buffers]

        assert self.blocks

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
        if hasattr(item, "__iter__"):
            item = tuple(item)
        else:
            item = (item,)

        ndim = len(self.shape)
        block_size = len(self.blocks[0])

        assert len(item) <= ndim

        if len(item) == ndim and all(isinstance(i, int) for i in item):
            coord = [i if i >= 0 else self.shape[x] + i for x, i in enumerate(item)]
            offset = sum(i * stride for i, stride in zip(coord, strides_for(self.shape)))
            return self.blocks[offset // block_size][offset % block_size]

        bounds, shape = slice_bounds(self.shape, item)

        # characterize the source tensor (this tensor)
        block_axis = block_axis_for(self.shape, block_size)

        # characterize the output tensor (the slice of this tensor)
        block_map_bounds = bounds[:block_axis]
        if isinstance(bounds[block_axis], slice):
            stride = self.blocks[0].shape[0]
            bound = bounds[block_axis]
            bound = slice(int(np.floor(bound.start / stride)), int(np.ceil(bound.stop / stride)))
        else:
            bound = bounds[block_axis] // self.block_map.shape[-1]

        block_map_bounds.append(bound)
        block_map = self.block_map[block_map_bounds]

        global_bound = bounds[block_axis]
        if isinstance(global_bound, slice):
            stride = self.blocks[0].shape[0]
            global_bound_size = (global_bound.stop - global_bound.start) // global_bound.step
            assert len(block_map) == np.ceil(global_bound_size / stride)
            if len(block_map) == 1:
                local_bounds = [global_bound]
            else:
                local_bounds = [slice(global_bound.start % stride, None, global_bound.step)]
                local_bounds += [slice(None, None, global_bound.step)] * (len(block_map) - 2)
                local_bounds += [slice(None, global_bound.stop % stride, global_bound.step)]
        else:
            # a single coordinate on the block axis should mean there is a single block selected
            assert len(block_map) == 1
            local_bounds = [global_bound]

        blocks = []
        for (i, bound) in zip(block_map, local_bounds):
            block = self.blocks[i]
            block_bounds = bounds[block_axis:]
            block_bounds[0] = bound
            blocks.append(block[block_bounds])

        return TensorView(blocks, block_map, shape)

    def __iter__(self):
        for i in self.block_map:
            for n in self.blocks[i]:
                yield n

    def __len__(self):
        return np.product(self.shape)

    def __setitem__(self, key, value):
        key = tuple(key)
        ndim = len(self.shape)
        block_axis = block_axis_for(self.shape, len(self.blocks[0]))

        if len(key) == ndim and all(isinstance(i, int) for i in key):
            coord = [i if i >= 0 else self.shape[x] + i for x, i in enumerate(key)]
            map_coord = coord[:block_axis]
            map_coord.append(coord[block_axis] // self.block_map.shape[-1])
            block_coord = coord[block_axis:]
            block_coord[0] = block_coord[0] // map_coord[-1]
            block = self.block_map[map_coord]
            block[block_coord] = value
            return

        bounds, shape = slice_bounds(self.shape, key)
        value = value.broadcast(shape)
        strides = strides_for(value.shape)

        for i, n in enumerate(value):
            source_coord = tuple((i // stride) % dim for dim, stride in zip(value.shape, strides))

            x = 0
            coord = []
            for bound in bounds:
                if isinstance(bound, int):
                    coord.append(bound)
                else:
                    coord.append(bound.start + (source_coord[x] * bound.step))
                    x += 1

            self[coord] = n

    def broadcast(self, shape):
        if self.shape == shape:
            return self

        # characterize the source tensor (this tensor)
        block_axis = block_axis_for(self.shape, len(self.blocks[0]))

        # characterize the output tensor (the broadcasted view of this tensor)
        shape = list(shape)
        offset = len(shape) - len(self.shape)
        block_shape = shape[offset + block_axis:]
        map_shape = shape[:offset + block_axis]

        block_map = self.block_map.broadcast(map_shape) if map_shape else self.block_map
        blocks = [self.blocks[i].broadcast(block_shape) for i in block_map]

        return TensorView(blocks, block_map, shape)

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

        if all(i == x for i, x in enumerate(permutation)):
            return self

        # characterize the source tensor (this tensor)
        ndim = len(self.shape)
        block_axis = block_axis_for(self.shape, len(self.blocks[0]))

        # characterize the output tensor (the transpose of this tensor)
        map_permutation = [x for x in permutation if x < block_axis] + [block_axis]
        block_map = self.block_map.transpose(map_permutation)

        block_permutation = [0] + [x - block_axis for x in permutation if x > block_axis]
        blocks = [block.transpose(block_permutation) for block in self.blocks]

        # return a view if possible
        shape = [self.shape[x] for x in map_permutation[:-1]] + [self.shape[block_axis + x] for x in block_permutation]
        view = TensorView(blocks, block_map, shape)

        view_axes = map_permutation[:-1] + [block_axis] + [block_axis + x for x in block_permutation[1:]]
        assert len(view_axes) == ndim and all(x in view_axes for x in range(ndim))
        view_permutation = [permutation.index(x) for x in view_axes]
        assert len(view_axes) == ndim and all(x in view_axes for x in range(ndim))
        assert permutation == [view_axes[x] for x in view_permutation]

        if all(i == x for i, x in enumerate(view_permutation)):
            return view

        # otherwise, construct a new tensor
        shape = [self.shape[x] for x in permutation]
        transpose = Tensor(shape)

        # and construct the remaining permutation of the view

        axes = [x for i, x in enumerate(view_permutation) if x != i]
        dims = [view.shape[x] for i, x in enumerate(view_permutation) if x != i]
        strides = strides_for(dims)

        # for each coordinate within the dimensions to transpose
        for o in range(int(np.product(dims))):
            coord = tuple((o // stride) % dim for dim, stride in zip(dims, strides))

            # read a slice of the view
            # transpose the slice
            # write it to the output tensor

        return transpose


class TensorView(Tensor):
    def __init__(self, blocks, block_map, shape):
        assert blocks
        assert isinstance(block_map, Block), f"invalid block map: {block_map}"
        assert shape

        self.blocks = blocks
        self.block_map = block_map
        self.shape = tuple(int(dim) for dim in shape)


def block_axis_for(shape, block_size):
    assert shape and all(shape) and all(dim > 0 for dim in shape)

    block_axis = len(shape) - 1
    while np.product(shape[block_axis:]) < block_size:
        block_axis -= 1

    return block_axis
