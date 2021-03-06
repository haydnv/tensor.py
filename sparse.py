import itertools
import numpy as np

from collections import OrderedDict

import transform

from btree.table import Index, Schema, Table
from dense import DenseTensor, sort_coords
from tensor import Tensor, affected, broadcast, validate_match, validate_slice, product


class SparseAddressor(object):
    def __init__(self, shape, dtype):
        self.dtype = dtype
        self.ndim = len(shape)
        self.shape = shape
        self.size = product(shape)

    def broadcast(self, shape):
        return SparseBroadcast(self, shape)

    def filled(self, match=None):
        raise NotImplementedError

    def filled_at(self, axes):
        raise NotImplementedError

    def filled_count(self, match=None):
        count = 0
        for _ in self.filled(match):
            count += 1

        return count

    def expand_dims(self, axis):
        return SparseExpansion(self, axis)

    def transpose(self, axes):
        return SparseTranspose(self, axes)


class SparseIdentity(SparseAddressor):
    def __init__(self, size, dtype):
        assert size == int(size) and size > 0
        SparseAddressor.__init__(self, [size, size], dtype)

    def __getitem__(self, match):
        match = validate_match(match, self.shape)
        if len(match) == 2 and all(isinstance(c, int) for c in match):
            if match[0] == match[1]:
                return self.dtype(1)
            else:
                return self.dtype(0)
        else:
            return SparseSlice(self, match)

    def filled(self, match=None):
        one = self.dtype(1)

        if match is None:
            for i in range(self.shape[0]):
                yield (i, i), one
            return

        match = validate_match(match, self.shape)
        for coord in itertools.product(*affected(match, self.shape)):
            assert len(coord) == 2
            if coord[0] == coord[1]:
                yield coord, one

    def filled_at(self, axes):
        if len(axes) == 1:
            yield from range(self.shape[0])
        elif len(axes) == 2:
            for i in range(self.shape([0])):
                yield (i, i)
        else:
            raise ValueError

    def filled_count(self, match=None):
        if match is None:
            return self.shape[0]
        else:
            count = 0
            for _ in self.filled(match):
                count += 1
            return count


class SparseTable(SparseAddressor):
    def __init__(self, shape, dtype, table=None):
        SparseAddressor.__init__(self, shape, dtype)

        if table is None:
            self.table = Table(Index(Schema(
                [(i, int) for i in range(self.ndim)],
                [("value", self.dtype)])))

            for i in range(self.ndim):
                self.table.add_index(str(i), [i])
        else:
            self.table = table

    def __getitem__(self, match):
        match = validate_match(match, self.shape)
        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            selector = dict(zip(range(len(match)), match))
            for (value,) in self.table.slice(selector).select(["value"]):
                return value

            return self.dtype(0)
        else:
            return SparseSlice(self, match)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)

        if isinstance(value, SparseTensor):
            dest = self[match]
            if dest.shape != value.shape:
                value = value.broadcast(dest.shape)

            for coord, val in value.filled():
                dest[coord] = val

        elif value == self.dtype(0):
            self._delete_filled(match)
        else:
            affected_range = affected(match, self.shape)

            if isinstance(value, Tensor):
                if value.shape != self[match].shape:
                    value = value.broadcast(self[match].shape)

                value = iter(value)
                for coord in itertools.product(*affected_range):
                    self.table.upsert(coord, (next(value),))
            else:
                value = self.dtype(value)
                for coord in itertools.product(*affected_range):
                    self.table.upsert(coord, (value,))

    def filled(self, match=None):
        table = self.table if match is None else slice_table(self.table, match, self.shape)
        for row in table:
            yield (tuple(row[:-1]), row[-1])

    def filled_at(self, axes, match=None):
        if match is None:
            table = self.table
        else:
            table = slice_table(self.table, match, self.shape)

        yield from self.table.group_by(axes)

    def filled_count(self, match=None):
        if match is None:
            return self.table.count()
        else:
            return slice_table(self.table, match, self.shape).count()

    def _delete_filled(self, match):
        slice_table(self.table, match, self.shape).delete()


class SparseRebase(SparseAddressor):
    def __init__(self, rebase, source):
        SparseAddressor.__init__(self, rebase.shape, source.dtype)
        self._source = source
        self._rebase = rebase

    def __getitem__(self, match):
        match = self._rebase.invert_coord(match)
        return self._source[match]

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)
        match = self._rebase.invert_coord(match)
        self._source[match] = value

    def filled(self, match=None):
        if match:
            match = self._rebase.invert_coord(match)

        for coord, value in self._source.filled(match):
            yield (self._rebase.map_coord(coord), value)

    def filled_count(self, match=None):
        if match:
            match = self._rebase._invert_coord(match)

        return self._source.filled_count(match)


class SparseBroadcast(SparseRebase):
    def __init__(self, source, broadcast_shape):
        source_shape = list(source.shape)
        offset = len(broadcast_shape) - len(source_shape)
        if offset:
            source_shape = ([1] * offset) + source_shape

        shape = []
        for l, r in zip(source_shape, broadcast_shape):
            if l == r or r == 1:
                shape.append(l)
            elif l == 1:
                shape.append(r)
            else:
                raise ValueError("cannot broadcast {} into {}".format(source.shape, broadcast_shape))

        rebase = transform.Broadcast(source.shape, shape)
        SparseRebase.__init__(self, rebase, source)

    def __getitem__(self, match):
        shape = []
        for axis in range(self.ndim):
            if axis < len(match):
                if isinstance(match[axis], int):
                    pass
                else:
                    shape.append(math.ceil((match.stop - match.start) / match.step))
            else:
                shape.append(self.shape[axis])

        match = self._rebase.invert_coord(match)
        value = self._source[match]
        if shape == []:
            return value
        else:
            return value.broadcast(shape)

    def _map_coords(self, match):
        for source_coord, _ in self._source.filled(match):
            match = self._rebase.map_coord(source_coord)
            for coord in itertools.product(*affected(match, self.shape)):
                yield coord

    def filled(self, match=None):
        match = None if match is None else self._rebase.invert_coord(match)

        coords = self._map_coords(match)
        coords = [
            tuple(int(c) for c in coord)
            for coord in sort_coords(coords, self.shape)
        ]
        assert coords == sorted(coords)

        for coord in coords:
            source_coord = self._rebase.invert_coord(coord)
            value = self._source[source_coord]
            assert value
            yield (coord, value)


class SparseCombine(SparseAddressor):
    def __init__(self, left, right, combinator, dtype):
        assert left.shape == right.shape

        self._left = left
        self._right = right
        self._combinator = combinator

        SparseAddressor.__init__(self, left.shape, dtype)

    def __getitem__(self, match):
        if len(match) == len(self.shape) and all(isinstance(c, int) for c in match):
            return self._combinator(self._left[match], self._right[match])
        else:
            return SparseCombine(
                self._left[match], self._right[match],
                self._combinator, self.dtype)

    def filled(self, match=None):
        assert list(self._left.filled(match)) == sorted(self._left.filled(match))
        assert list(self._right.filled(match)) == sorted(self._right.filled(match))

        left = self._left.filled(match)
        right = self._right.filled(match)

        left_done = False
        right_done = False

        left_next = None
        right_next = None

        coord_index = np.array(
            [product(self.shape[axis + 1:]) for axis in range(len(self.shape))])

        while True:
            if left_next is None:
                try:
                    left_next = next(left)
                except StopIteration:
                    left_done = True

            if right_next is None:
                try:
                    right_next = next(right)
                except StopIteration:
                    right_done = True

            if left_done and right_next:
                value = self._combinator(self._left.dtype(0), right_next[1])
                if value:
                    yield (right_next[0], value)
                    right_next = None

            if right_done and left_next:
                value = self._combinator(left_next[1], self._right.dtype(0))
                if value:
                    yield (left_next[0], value)
                    left_next = None

            if left_done or right_done:
                break
            else:
                (left_coord, left_value) = left_next
                (right_coord, right_value) = right_next
                left_index = np.sum(np.array(left_coord) * coord_index)
                right_index = np.sum(np.array(right_coord) * coord_index)

                if left_index == right_index:
                    left_next = None
                    right_next = None

                    value = self._combinator(left_value, right_value)
                    if value:
                        yield (left_coord, value)

                elif left_index < right_index:
                    left_next = None

                    value = self._combinator(left_value, self._right.dtype(0))
                    if value:
                        yield (left_coord, value)

                elif right_index < left_index:
                    right_next = None

                    value = self._combinator(self._left.dtype(0), right_value)
                    if value:
                        yield (right_coord, value)

                else:
                    raise RuntimeError

        if left_done and not right_done:
            for coord, right_value in right:
                value = self._combinator(self._left.dtype(0), right_value)
                if value:
                    yield (coord, value)

        elif right_done and not left_done:
            for coord, left_value in left:
                value = self._combinator(left_value, self._right.dtype(0))
                if value:
                    yield (coord, value)

    def filled_at(self, axes):
        if axes == sorted(axes):
            group = None
            for coord, _ in self.filled():
                filled_at = tuple(coord[axis] for axis in axes)
                if filled_at != group:
                    yield filled_at
                    group = filled_at
        else:
            raise NotImplementedError


class SparseExpansion(SparseRebase):
    def __init__(self, source, axis):
        rebase = transform.Expand(source.shape, axis)
        SparseRebase.__init__(self, rebase, source)


class SparseSlice(SparseRebase):
    def __init__(self, source, match):
        self._match = validate_match(match, source.shape)
        rebase = transform.Slice(source.shape, self._match)
        SparseRebase.__init__(self, rebase, source)

    def filled(self, match=None):
        if match:
            match = self._rebase.invert_coord(match)
        else:
            match = self._match

        for coord, value in self._source.filled(match):
            yield (self._rebase.map_coord(coord), value)

    def filled_at(self, axes, match=None):
        if match:
            match = self._rebase.invert_coord(match)
        else:
            match = self._match

        axes = self._rebase.invert_axes(axes)
        yield from self._source.filled_at(axes, match)

    def filled_count(self, match=None):
        if match:
            match = self._rebase.invert_coord(match)
        else:
            match = self._match

        return self._source.filled_count(match)


class SparseTranspose(SparseRebase):
    def __init__(self, source, permutation=None):
        rebase = transform.Transpose(source.shape, permutation)
        SparseRebase.__init__(self, rebase, source)

    def __getitem__(self, coord):
        source = self._source[self._rebase.invert_coord(coord)]
        if not hasattr(source, "shape") or source.shape == tuple():
            return source

        permutation = OrderedDict(zip(range(self.ndim), self._rebase.permutation))
        elided = []
        for axis in range(len(coord)):
            if isinstance(coord[axis], int):
                elided.append(permutation[axis])
                del permutation[axis]

        for axis in elided:
            for i in permutation:
                if permutation[i] > axis:
                    permutation[i] -= 1

        return source.transpose(list(permutation.values()))

    def filled(self, match=None):
        if match:
            match = self._rebase.invert_coord(match)

        for (coord, value) in self._source.filled(match):
            yield (self._rebase.map_coord(coord), value)

    def filled_at(self, axes, match=None):
        axes = self._rebase.invert_axes(axes)
        if match:
            match = self._rebase.invert_coord(match)

        for source_coord in self._source.filled_at(axes, match):
            dest_coord = self._rebase.map_coord_axes(source_coord, axes)
            yield dest_coord


class SparseTensor(Tensor):
    @staticmethod
    def identity(size, dtype=np.int32):
        accessor = SparseIdentity(size, dtype)
        return SparseTensor(accessor.shape, dtype, accessor)

    def __init__(self, shape, dtype=np.int32, accessor=None):
        Tensor.__init__(self, shape, dtype)
        if accessor is None:
            self.accessor = SparseTable(shape, dtype)
        else:
            assert shape == accessor.shape
            assert dtype == accessor.dtype
            self.accessor = accessor

    def __add__(self, other):
        if not isinstance(other, SparseTensor):
            raise NotImplemented

        this, that = broadcast(self, other)

        accessor = SparseCombine(this.accessor, that.accessor, lambda l, r: l + r, this.dtype)
        return SparseTensor(accessor.shape, accessor.dtype, accessor)

    def __and__(self, other):
        if not isinstance(other, SparseTensor):
            raise NotImplemented

        this, that = broadcast(self, other)

        accessor = SparseCombine(this.accessor, that.accessor, lambda l, r: l and r, np.bool)
        return SparseTensor(accessor.shape, np.bool, accessor)

    def __eq__(self, other):
        if isinstance(other, DenseTensor):
            return other == self
        elif isinstance(other, SparseTensor):
            left = other if other.shape == self.shape else other.broadcast(self.shape)
            right = self if self.shape == other.shape else self.broadcast(other.shape)
            eq = other == self.dtype(0)
            for coord, value in self.filled():
                eq[coord] = other[coord] == value

            return eq
        else:
            eq = DenseTensor.ones(self.shape, np.bool) * (other == self.dtype(0))
            for coord, value in self.filled():
                eq[coord] = other == value

            return eq

    def __getitem__(self, match):
        match = validate_match(match, self.shape)
        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            return self.accessor[match]
        else:
            slice_accessor = self.accessor[match]
            return SparseTensor(slice_accessor.shape, self.dtype, slice_accessor)

    def __mul__(self, other):
        if not isinstance(other, Tensor) and np.array(other).shape == tuple():
            multiplied = SparseTensor(self.shape, self.dtype)
            for coord, value in self.filled():
                multiplied[coord] = value * other
            return multiplied

        this, that = broadcast(self, other)

        multiplied = SparseTensor(this.shape, this.dtype)
        if isinstance(that, SparseTensor) and that.filled_count() < this.filled_count():
            for coord, value in that.filled():
                multiplied[coord] = value * this[coord]
        else:
            for coord, value in this.filled():
                multiplied[coord] = value * that[coord]

        return multiplied

    def __or__(self, other):
        if not isinstance(other, SparseTensor):
            raise NotImplemented

        this, that = broadcast(self.accessor, other.accessor)

        accessor = SparseCombine(this, that, lambda l, r: l or r, np.bool)
        return SparseTensor(this.shape, np.bool, accessor)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)
        self.accessor[match] = value

    def all(self):
        return self.accessor.filled_count() == self.size

    def any(self):
        return self.accessor.filled_count() > 0

    def broadcast(self, shape):
        if shape == self.shape:
            return self

        accessor = SparseBroadcast(self.accessor, shape)
        return SparseTensor(accessor.shape, self.dtype, accessor)

    def copy(self):
        copy = SparseTensor(self.shape, self.dtype)
        for (coord, value) in self.filled():
            copy[coord] = value

        return copy

    def expand(self, new_shape):
        if not isinstance(self.accessor, SparseTable):
            raise NotImplementedError

        if len(new_shape) != len(self.shape):
            return ValueError
        elif not (np.array(new_shape) >= np.array(self.shape)).all():
            return ValueError

        self.accessor.shape = new_shape
        self.accessor.size = product(new_shape)

        self.shape = self.accessor.shape
        self.size = self.accessor.size

    def expand_dims(self, axis):
        accessor = self.accessor.expand_dims(axis)
        return SparseTensor(accessor.shape, self.dtype, accessor)

    def filled(self):
        yield from self.accessor.filled()

    def filled_at(self, axes):
        yield from self.accessor.filled_at(axes, None)

    def filled_count(self):
        return self.accessor.filled_count()

    def mask(self, other):
        other = other.broadcast(self.shape)

        if isinstance(other, SparseTensor):
            for coord, _ in other.filled():
                self[coord] = self.dtype(0)
        elif isinstance(other, Tensor):
            for coord in itertools.product(*[range(dim) for dim in other.shape]):
                if other[coord]:
                    self[coord] = self.dtype(0)
        else:
            raise ValueError

    def product(self, axis = None):
        if axis is None or (axis == 0 and self.ndim == 1):
            if self.all():
                return product(value for _, value in self.filled())
            else:
                return self.dtype(0)

        assert axis < self.ndim
        shape = list(self.shape)
        del shape[axis]
        multiplied = SparseTensor(shape, self.dtype)

        if axis == 0:
            for coord in self.filled_at(list(range(1, self.ndim))):
                multiplied[coord] = self[(slice(None),) + coord].product()
        else:
            for prefix in self.filled_at(list(range(axis))):
                multiplied[prefix] = self[prefix].product(0)

        return multiplied

    def sum(self, axis = None):
        if axis is None or (axis == 0 and self.ndim == 1):
            return sum(value for _, value in self.filled())

        assert axis < self.ndim
        shape = list(self.shape)
        del shape[axis]
        summed = SparseTensor(shape, self.dtype)

        if axis == 0:
            for coord in self.filled_at(list(range(1, self.ndim))):
                summed[coord] = self[(slice(None),) + coord].sum()
        else:
            for prefix in self.filled_at(list(range(axis))):
                summed[prefix] = self[prefix].sum(0)

        return summed

    def transpose(self, permutation=None):
        if permutation == list(range(self.ndim)):
            return self

        accessor = self.accessor.transpose(permutation)
        return SparseTensor(accessor.shape, self.dtype, accessor)

    def to_nparray(self):
        dense = np.zeros(self.shape, self.dtype)
        for (coord, value) in self.filled():
            dense[coord] = value

        return dense


def slice_table(table, match, shape):
    selector = {}
    steps = {}
    for axis in range(len(match)):
        coord = match[axis]
        if isinstance(coord, slice):
            coord = validate_slice(coord, shape[axis])
            default = validate_slice(slice(None), shape[axis])
            if coord == default:
                pass
            elif coord.step > 1:
                selector[axis] = slice(coord.start, coord.stop, 1)
                if coord.step != 1:
                    steps[axis] = (coord.start, coord.step)
            else:
                selector[axis] = coord
        else:
            selector[axis] = coord

    table_slice = table.slice(selector)

    if steps:
        def step_filter(row):
            step_match = all(
                (row[axis] - offset) % step == 0
                for (axis, (offset, step)) in steps.items())
            return step_match

        table_slice = table_slice.filter(step_filter)

    return table_slice

