import itertools
import math
import numpy as np

from btree.table import Index, Schema, Table
from base import Broadcast, Expansion, Permutation, Tensor, TensorSlice
from base import affected, product, validate_match, validate_slice, validate_tuple
from dense import BlockTensor


class SparseTensorView(Tensor):
    def __init__(self, shape, dtype):
        super().__init__(shape, dtype)

    def __eq__(self, other):
        if isinstance(other, self.dtype):
            eq = BlockTensor.ones(self.shape, np.bool) * (other == self.dtype(0))
            for row in self.filled():
                eq[row[:-1]] = row[-1] == other
            return eq
        elif self.shape != other.shape:
            shape = [max(l, r) for l, r in zip(self.shape, other.shape)]
            return self.broadcast(shape) == other.broadcast(shape)
        elif isinstance(other, Tensor):
            eq = other == other.dtype(0)
            for row in self.filled():
                eq[row[:-1]] = row[-1] == other[row[:-1]]
            return eq
        else:
            raise ValueError

    def __mul__(self, other):
        if not isinstance(other, Tensor) and np.array(other).shape == tuple():
            multiplied = SparseTensor(self.shape)
            for row in self.filled():
                multiplied[row[:-1]] = self[row[:-1]] * other
            return multiplied

        shape = [max(l, r) for l, r in zip(self.shape, other.shape)]
        this = self.broadcast(shape)
        that = other.broadcast(shape)

        multiplied = SparseTensor(this.shape, this.dtype)
        if isinstance(that, SparseTensorView) and that.filled_count() < this.filled_count():
            for row in that.filled():
                coord = row[:-1]
                multiplied[coord] = this[coord] * that[coord]
        else:
            for row in this.filled():
                coord = row[:-1]
                multiplied[coord] = this[coord] * that[coord]

        return multiplied

    def __sub__(self, other):
        if isinstance(other, SparseTensorView):
            left = self
            right = other

            if left.dtype == np.bool:
                left = left.as_type(np.uint8)

            if right.dtype == np.bool:
                right = right.as_type(np.uint8)

            if left.shape == right.shape:
                pass
            elif right.ndim > left.ndim:
                left = left.broadcast(right.shape)
            else:
                right = right.broadcast(left.shape)

            subtraction = SparseTensor(left.shape, left.dtype)

            for row in left.filled():
                coord = row[:-1]
                val = row[-1]
                subtraction[coord] = val - right[coord]

            for row in right.filled():
                coord = row[:-1]
                val = row[-1]
                subtraction[coord] = left[coord] - val

            return subtraction
        else:
            Tensor.__sub__(self, other)

    def __xor__(self, other):
        if isinstance(other, SparseTensorView):
            if other.ndim > self.ndim:
                return other ^ self

            this = bool(self)
            that = bool(other)

            if this.shape != that.shape:
                that = that.broadcast(self.shape)

            xor = SparseTensor(this.shape, np.bool)

            for row in this.filled():
                coord = row[:-1]
                val = row[-1]
                xor[coord] = val ^ that[coord]

            for row in that.filled():
                coord = row[:-1]
                val = row[-1]
                xor[coord] = this[coord] ^ val

            return xor
        else:
            Tensor.__xor__(self, other)

    def all(self):
        if not self.shape:
            return True

        expected = product(self.shape)
        actual = 0
        for _ in self.filled():
            actual += 1

        return expected == actual

    def any(self):
        for row in self.filled():
            return True

        return False

    def as_type(self, cast_to):
        if cast_to == self.dtype:
            return self

        return self._copy(cast_to)

    def broadcast(self, shape):
        if shape == self.shape:
            return self

        return SparseBroadcast(self, shape)

    def copy(self):
        return self._copy(self.dtype)

    def expand(self, new_shape):
        assert len(new_shape) == self.ndim
        for axis in range(self.ndim):
            assert new_shape[axis] >= self.shape[axis]

        self.shape = new_shape
        self.size = product(new_shape)

    def filled(self):
        raise NotImplementedError

    def filled_count(self):
        raise NotImplementedError

    def product(self, axis):
        assert axis < self.ndim

        shape = list(self.shape)
        del shape[axis]
        product = SparseTensor(shape, self.dtype)
        visited = SparseTensor(shape, np.bool)

        for row in self.filled():
            elide = list(row[:-1])
            elide[axis] = slice(None)
            elide = tuple(elide)
            if not self[elide].all():
                continue

            if axis == 0:
                coord = row[1:-1]
                if visited[coord]:
                    product[coord] *= row[-1]
                else:
                    visited[coord] = True
                    product[coord] = row[-1]
            elif axis == self.ndim - 1:
                coord = row[:-2]
                if visited[coord]:
                    product[coord] *= row[-1]
                else:
                    visited[coord] = True
                    product[coord] = row[-1]
            else:
                l = row[:axis]
                r = row[(axis + 1):-1]
                if visited[l][r]:
                    product[l][r] *= row[-1]
                else:
                    visited[l][r] = True
                    product[l][r] = row[-1]

        return product

    def sum(self, axis):
        assert axis < self.ndim

        shape = list(self.shape)
        del shape[axis]
        summed = SparseTensor(shape, self.dtype)

        if axis == 0:
            for row in self.filled():
                summed[row[1:-1]] += row[-1]
        elif axis == self.ndim - 1:
            for row in self.filled():
                summed[row[:-2]] += row[-1]
        else:
            for row in self.filled():
                coord = row[:-1]
                val = row[-1]
                summed[coord[:axis]][coord[(axis + 1):]] += val

        return summed

    def transpose(self, permutation=None):
        return SparsePermutation(self, permutation)

    def to_nparray(self):
        dense = np.zeros(self.shape, self.dtype)
        for entry in self.filled():
            coord = entry[:-1]
            value = entry[-1]
            dense[coord] = value

        return dense

    def _copy(self, dtype):
        copied = SparseTensor(self.shape, dtype)
        for row in self.filled():
            coord = row[:-1]
            copied[coord] = dtype(row[-1])

        return copied


class SparseTensor(SparseTensorView):
    def __init__(self, shape, dtype=np.int32):
        super().__init__(tuple(shape), dtype)

        self._table = Table(Index(Schema(
            [(i, int) for i in range(self.ndim)],
            [("value", self.dtype)])))

        for i in range(self.ndim):
            self._table.add_index(str(i), [i])

    def __getitem__(self, match):
        match = validate_match(match, self.shape)

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            if any(abs(match[axis]) > self.shape[axis] for axis in range(self.ndim)):
                raise IndexError

            match = [
                match[i] if match[i] >= 0 else self.shape[i] + match[i]
                for i in range(self.ndim)]
            selector = dict(zip(range(len(match)), match))
            for (value,) in self._table.slice(selector).select(["value"]):
                return value

            return self.dtype(0)
        else:
            return SparseTensorSlice(self, match)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)

        if isinstance(value, SparseTensorView):
            dest = self[match]
            if dest.shape != value.shape:
                value = value.broadcast(dest.shape)

            for row in value.filled():
                dest[row[:-1]] = row[-1]

        elif isinstance(value, Tensor):
            Tensor.__setitem__(self, match, value)
        elif value == self.dtype(0):
            self._delete_filled(match)
        else:
            affected_range = affected(match, self.shape)

            value = self.dtype(value)
            for coord in itertools.product(*affected_range):
                self._table.upsert(coord, (value,))

            self._table.rebalance()

    def expand_dims(self, axis):
        return SparseExpansion(self, axis)

    def filled(self, match=None):
        if match is None:
            yield from self._table
        else:
            yield from self._slice_table(match)

    def filled_count(self):
        return len(self._table)

    def _delete_filled(self, match):
        self._slice_table(match).delete()

    def _slice_table(self, match):
        selector = {}
        steps = {}
        for axis in range(len(match)):
            coord = match[axis]
            if isinstance(coord, slice):
                coord = validate_slice(coord, self.shape[axis])
                default = validate_slice(slice(None), self.shape[axis])
                if coord == default:
                    pass
                elif coord.step != 1:
                    selector[axis] = slice(coord.start, coord.stop, 1)
                    if coord.step != 1:
                        steps[axis] = (coord.start, coord.step)
                else:
                    selector[axis] = coord
            else:
                selector[axis] = coord

        table_slice = self._table.slice(selector)
        if steps:
            def step_filter(row):
                step_match = all(
                    (row[axis] - offset) % step == 0
                    for (axis, (offset, step)) in steps.items())
                return step_match

            table_slice = table_slice.filter(step_filter)

        return table_slice


class SparseRebase(SparseTensorView):
    def __init__(self, source, shape):
        super().__init__(shape, source.dtype)
        self._source = source

    def filled(self):
        for row in self._source.filled():
            coord = row[:-1]
            value = row[-1]
            yield self._map_coord(coord) + (value,)

    def filled_count(self):
        return self._source.filled_count()


class SparseBroadcast(Broadcast, SparseRebase):
    def __init__(self, source, shape):
        Broadcast.__init__(self, source, shape)
        SparseRebase.__init__(self, source, shape)

    def filled(self):
        for row in self._source.filled():
            coord = self._map_coord(row[:-1])
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
                yield tuple(broadcast_coord) + (row[-1],)

class SparseExpansion(Expansion, SparseRebase):
    def __init__(self, source, axis):
        Expansion.__init__(self, source, axis)
        SparseRebase.__init__(self, source, self.shape)


class SparsePermutation(Permutation, SparseRebase):
    def __init__(self, source, permutation=None):
        Permutation.__init__(self, source, permutation)
        SparseRebase.__init__(self, source, self.shape)


class SparseTensorSlice(TensorSlice, SparseRebase):
    def __init__(self, source, match):
        assert isinstance(source, SparseTensor)

        TensorSlice.__init__(self, source, match)
        SparseRebase.__init__(self, source, self.shape)

    def filled(self):
        for row in self._source.filled(self._match):
            coord = row[:-1]
            value = row[-1]
            yield self._map_coord(coord) + (value,)

    def filled_count(self):
        return self._source.filled(self._match).count()

