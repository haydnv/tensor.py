import itertools
import math
import numpy as np

from btree.table import Index, Schema, Table
from base import Broadcast, Tensor


class SparseTensorView(Tensor):
    def __init__(self, shape, dtype, default):
        super().__init__(shape, dtype)
        self._default = default

    def broadcast(self, shape):
        return SparseBroadcast(self, shape)

    def to_dense(self):
        dense = np.ones(self.shape, self.dtype) * self._default
        for entry in self.filled():
            coord = entry[:-1]
            value = entry[-1]
            dense[coord] = value

        return dense


class SparseTensor(SparseTensorView):
    def __init__(self, shape, dtype=np.int32, default=0):
        super().__init__(tuple(shape), dtype, default)

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

            return self._default
        else:
            return SparseTensorSlice(self, match)

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)

        if isinstance(value, SparseTensorView) and value._default == self._default:
            dest = self[match]
            if dest.shape != value.shape:
                value = value.broadcast(dest.shape)

            for row in value.filled():
                dest[row[:-1]] = row[-1]

        elif isinstance(value, Tensor):
            Tensor.__setitem__(self, match, value)
        elif value == self._default:
            self._delete_filled(match)
        else:
            affected = []
            for axis in range(len(match)):
                if match[axis] is None:
                    affected.append(range(self.shape[axis]))
                elif isinstance(match[axis], slice):
                    s = match[axis]
                    affected.append(range(s.start, s.stop, s.step))
                elif isinstance(match[axis], tuple):
                    affected.append(match[axis])
                elif match[axis] < 0:
                    affected.append([self.shape[axis] + match[axis]])
                else:
                    affected.append([match[axis]])

            for axis in range(len(match), self.ndim):
                affected.append(range(self.shape[axis]))

            value = self.dtype(value)
            for coord in itertools.product(*affected):
                self._table.upsert(coord, (value,))

            self._table.rebalance()

    def __sub__(self, other):
        raise NotImplementedError

    def __xor__(self, other):
        if isinstance(other, SparseTensorView):
            if other.ndim > self.ndim:
                return other ^ self

            this = bool(self)
            that = bool(other)

            if this.shape != that.shape:
                that = that.broadcast(self.shape)

            xor = SparseTensor(this.shape, np.bool, (this._default ^ that._default))

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
            Tensor.__xor__(this, that)

    def as_type(self, cast_to):
        if cast_to == self.dtype:
            return self

        return self._copy(cast_to)

    def copy(self):
        return self._copy(self.dtype)

    def filled(self, match=None):
        if match is None:
            yield from self._table
        else:
            yield from self._slice_table(match)

    def _copy(self, dtype):
        copied = SparseTensor(self.shape, dtype, dtype(self._default))
        for row in self.filled():
            coord = row[:-1]
            copied[coord] = dtype(row[-1])

        return copied

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


class SparseBroadcast(Broadcast, SparseTensorView):
    def __init__(self, source, shape):
        Broadcast.__init__(self, source, shape)
        SparseTensorView.__init__(self, shape, source.dtype, source._default)

    def filled(self):
        for row in self._source.filled():
            coord = row[:-1]
            value = row[-1]
            yield self._map_coord(coord) + (value,)


class SparseTensorSlice(SparseTensorView):
    def __init__(self, source, match):
        match = validate_match(match, source.shape)

        shape = []
        offset = {}
        elided = []

        for axis in range(len(match)):
            if match[axis] is None:
                shape.append(source.shape[axis])
                offset[axis] = 0
            elif isinstance(match[axis], slice):
                s = match[axis]
                shape.append(math.ceil((s.stop - s.start) / s.step))
                offset[axis] = s.start
            elif isinstance(match[axis], tuple):
                shape.append(len(match[axis]))
            else:
                elided.append(axis)

        for axis in range(len(match), source.ndim):
            shape.append(source.shape[axis])
            offset[axis] = 0

        super().__init__(tuple(shape), source.dtype, source._default)
        self._source = source
        self._match = match

        def map_coord(source_coord):
            assert len(source_coord) == source.ndim
            dest_coord = []
            for axis in range(source.ndim):
                if axis in elided:
                    pass
                elif isinstance(source_coord[axis], slice):
                    raise NotImplementedError
                elif isinstance(source_coord[axis], tuple):
                    dest_coord.append(tuple(c - offset[axis] for c in source_coord[axis]))
                else:
                    dest_coord.append(source_coord[axis] - offset[axis])

            for axis in range(len(source_coord), self.ndim):
                if not axis in elided:
                    dest_coord.append(None)

            assert len(dest_coord) == self.ndim
            return tuple(dest_coord)

        def invert_coord(coord):
            coord = [
                coord[i] if i < len(coord) else slice(None)
                for i in range(self.ndim)]

            source_coord = []
            for axis in range(source.ndim):
                if axis in elided:
                    source_coord.append(match[axis])
                    continue

                at = coord.pop(0)
                if at is None:
                    if axis < len(match):
                        source_coord.append(match[axis])
                    else:
                        source_coord.append(None)
                elif isinstance(at, slice):
                    if axis < len(match):
                        match_axis = match[axis]
                    else:
                        match_axis = validate_slice(slice(None), source.shape[axis])

                    if isinstance(match_axis, slice):
                        if at.start is None:
                            start = match_axis.start
                        elif at.start < 0:
                            start = match_axis.stop + at.start
                        else:
                            start = at.start + match_axis.start

                        if at.step is None or at.step == 1:
                            step = match_axis.step
                        else:
                            step = match_axis.step * at.step

                        if at.stop is None:
                            stop = match_axis.stop
                        else:
                            at = validate_slice(at, (match_axis.stop - match_axis.start))
                            stop = start + (match_axis.step * (at.stop - at.start))

                        source_coord.append(slice(start, stop, step))
                    else:
                        if at.start != 0:
                            raise IndexError

                        source_coord.append(match[axis])
                elif isinstance(at, tuple):
                    source_coord.append(tuple(c + offset[axis] for c in at))
                else:
                    source_coord.append(at + offset[axis])

            return tuple(source_coord)

        self._map_coord = map_coord
        self._invert_coord = invert_coord

    def __getitem__(self, match):
        match = validate_match(match, self.shape)
        source_coord = self._invert_coord(match)
        return self._source[source_coord]

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)
        source_coord = self._invert_coord(match)
        self._source[source_coord] = value

    def filled(self):
        for row in self._source.filled(self._match):
            coord = row[:-1]
            value = row[-1]
            yield self._map_coord(coord) + (value,)


def validate_match(match, shape):
    if not isinstance(match, tuple):
        match = (match,)

    assert len(match) <= len(shape)

    match = list(match)
    for axis in range(len(match)):
        if match[axis] is None:
            pass
        elif isinstance(match[axis], slice):
            match[axis] = validate_slice(match[axis], shape[axis])
        elif isinstance(match[axis], tuple) or isinstance(match[axis], list):
            match[axis] = validate_tuple(match[axis], shape[axis])
        elif match[axis] < 0:
            assert abs(match[axis]) < shape[axis]
        else:
            assert match[axis] < shape[axis]

    return tuple(match)


def validate_slice(s, dim):
    if s.start is None:
        start = 0
    elif s.start < 0:
        start = dim + s.start
    else:
        start = s.start

    if s.stop is None:
        stop = dim
    elif s.stop < 0:
        stop = dim + s.stop
    else:
        stop = s.stop

    step = s.step if s.step else 1

    return slice(start, stop, step)


def validate_tuple(t, dim):
    if not t:
        raise IndexError

    if not all(isinstance(t[i], int) for i in range(len(t))):
        raise IndexError

    if any([abs(t[i]) > dim for i in range(len(t))]):
        raise IndexError

    return tuple(t)

