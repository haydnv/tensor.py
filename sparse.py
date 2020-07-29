import itertools
import numpy as np

import transform

from base import Tensor, affected, validate_match, validate_slice, product
from btree.table import Index, Schema, Table
from dense import DenseTensor


class SparseAddressor(object):
    def __init__(self, shape, dtype):
        self.dtype = dtype
        self.ndim = len(shape)
        self.shape = shape
        self.size = product(shape)


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
            return SparseTableSlice(self, match)

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

    def filled_at(self, axes):
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

    def filled_at(self, axes):
        group = None
        for coord, _ in self.filled():
            filled = tuple(coord[x] for x in axes)
            if filled != group:
                group = filled
                yield filled

    def filled_count(self, match=None):
        if match:
            match = self._rebase._invert_coord(match)

        return self._source.filled_count(match)
    

class SparseBroadcast(SparseRebase):
    def __init__(self, source, shape):
        rebase = transform.Broadcast(source.shape, shape)
        SparseRebase.__init__(self, rebase, source)

    def filled(self, match=None):
        if match:
            match = self._invert_coord(match)

        for coord, value in self._source.filled():
            coord = self._rebase.map_coord(coord)
            coord_range = []
            for axis in range(self.ndim):
                c = coord[axis]
                if isinstance(c, slice):
                    c = validate_slice(c, self.shape[axis])
                    coord_range.append(range(c.start, c.stop, c.step))
                elif isinstance(c, tuple):
                    coord_range.append(list(c))
                else:
                    coord_range.append([c])

            for broadcast_coord in itertools.product(*coord_range):
                yield (tuple(broadcast_coord), value)


class SparseTableSlice(SparseRebase):
    def __init__(self, source, match):
        self._match = validate_match(match, source.shape)
        rebase = transform.Slice(source.shape, self._match)
        SparseRebase.__init__(self, rebase, source)

    def filled(self, match=None):
        if match:
            match = self._invert_coord(match)
        else:
            match = self._match

        for coord, value in self._source.filled(match):
            yield (self._rebase.map_coord(coord), value)

    def filled_count(self, match=None):
        if match:
            match = self._invert_coord(match)
        else:
            match = self._match

        return self._source.filled_count(match)


class SparseTensor(Tensor):
    def __init__(self, shape, dtype=np.int32, accessor=None):
        Tensor.__init__(self, shape, dtype)
        if accessor is None:
            self.accessor = SparseTable(shape, dtype)
        else:
            assert shape == accessor.shape
            assert dtype == accessor.dtype
            self.accessor = accessor

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

        shape = [max(l, r) for l, r in zip(self.shape, other.shape)]
        this = self.broadcast(shape)
        that = other.broadcast(shape)

        multiplied = SparseTensor(this.shape, this.dtype)
        if isinstance(that, SparseTensor) and that.filled_count() < this.filled_count():
            for coord, value in that.filled():
                multiplied[coord] = value * this[coord]
        else:
            for coord, value in this.filled():
                multiplied[coord] = value * that[coord]

        return multiplied

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)
        self.accessor[match] = value

    def all(self):
        return self.accessor.filled_count() == self.size

    def any(self):
        return self.accessor.filled_count() > 0

    def broadcast(self, shape):
        accessor = SparseBroadcast(self.accessor, shape)
        return SparseTensor(accessor.shape, self.dtype, accessor)

    def filled(self):
        yield from self.accessor.filled()

    def filled_at(self, axes):
        yield from self.accessor.filled_at(axes)

    def filled_count(self):
        return self.accessor.filled_count()

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

