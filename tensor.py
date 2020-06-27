import itertools
import numpy as np

from btree.table import Index, Schema, Table


class Tensor(object):
    def __init__(self, shape):
        if not all(isinstance(dim, int) for dim in shape):
            raise ValueError

        if any(dim < 0 for dim in shape):
            raise ValueError

        self.shape = tuple(shape)
        self.ndim = len(shape)

    def __getitem__(self, _match):
        raise NotImplementedError

    def __setitem__(self, _match, _value):
        raise NotImplementedError

    def filled(self, match=None):
        raise NotImplementedError


class BlockTensor(Tensor):
    def __init__(self, source):
        super().__init__(source.shape)
        self._source = source

    def filled(self, match=None):
        source = self._source if match is None else self._source[match]
        for coord in itertools.product(*source.shape):
            value = source[coord]
            if value != 0:
                yield value


class SparseTensorView(Tensor):
    def __init__(self, shape, dtype=np.int32):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)

    def to_dense(self):
        dense = np.zeros(self.shape, self.dtype)
        for entry in self.filled():
            coord = entry[:-1]
            value = entry[-1]
            dense[coord] = value

        return dense


class SparseTensor(SparseTensorView):
    def __init__(self, shape, dtype=np.int32):
        super().__init__(shape, dtype)

        self._table = Table(Index(Schema(
            [(i, int) for i in range(self.ndim)],
            [("value", self.dtype)])))
        for i in range(self.ndim):
            self._table.add_index(str(i), [i])

    def __getitem__(self, match):
        if len(match) > self.ndim:
            raise IndexError

        if len(match) == self.ndim and all(isinstance(c, int) for c in match):
            if any(abs(match[axis]) > self.shape[axis] for axis in range(self.ndim)):
                raise IndexError

            match = [
                match[i] if match[i] >= 0 else self.shape[i] + match[i]
                for i in range(self.ndim)]
            selector = dict(zip(range(len(match)), match))
            for (value,) in self._table.slice(selector).select(["value"]):
                return value

            return 0

        return SparseTensorSlice(self, match)

    def __setitem__(self, match, value):
        if not isinstance(match, tuple):
            match = (match,)

        if len(match) > self.ndim:
            raise IndexError

        if isinstance(value, Tensor):
            dest = self[match]

            if value.ndim > dest.ndim:
                raise ValueError

            broadcast = [True for _ in range(dest.ndim)]
            offset = dest.ndim - value.ndim
            for axis in range(value.ndim):
                if dest.shape[offset + axis] == value.shape[axis]:
                    broadcast[offset + axis] = False
                elif dest.shape[offset + axis] != 1 and value.shape[axis] == 1:
                    broadcast[offset + axis] = True
                else:
                    print(dest.shape, value.shape, axis, offset)
                    raise ValueError("cannot broadcast")

            for row in value.filled():
                source_coord = row[:-1]
                val = row[-1]
                dest_coord = [None for _ in range(dest.ndim)]
                for axis in range(value.ndim):
                    if not broadcast[axis + offset]:
                        dest_coord[axis + offset] = source_coord[axis]

                dest[dest_coord] = val
        else:
            affected = []
            for axis in range(len(match)):
                if match[axis] is None:
                    affected.append(range(self.shape[axis]))
                elif isinstance(match[axis], slice):
                    s = validate_slice(match[axis], self.shape[axis])
                    affected.append(range(s.start, s.stop, s.step))
                elif isinstance(match[axis], tuple):
                    affected.append(validate_tuple(match[axis], self.shape[axis]))
                elif match[axis] < 0:
                    assert abs(match[axis]) < self.shape[axis]
                    affected.append([self.shape[axis] + match[axis]])
                else:
                    assert match[axis] < self.shape[axis]
                    affected.append([match[axis]])

            for axis in range(len(match), self.ndim):
                affected.append(range(self.shape[axis]))

            value = self.dtype(value)
            for coord in itertools.product(*affected):
                if value:
                    self._table.upsert(coord, (value,))
                else:
                    coord = dict(zip(range(self.ndim), coord))
                    self._table.slice(coord).delete()

            self._table.rebalance()

    def filled(self, match=None):
        if match is None:
            yield from self._table
        else:
            selector = dict(zip(range(len(match)), match))
            yield from self._table.slice(selector)


class SparseTensorSlice(SparseTensorView):
    def __init__(self, source, match):
        if not isinstance(match, tuple):
            match = (match,)

        shape = []
        offset = {}
        step = {}
        elided = []

        for axis in range(len(match)):
            if match[axis] is None:
                shape.append(source.shape[axis])
                offset[axis] = 0
            elif isinstance(match[axis], slice):
                s = validate_slice(match[axis], source.shape[axis])
                shape.append((s.stop - s.start) // s.step)
                offset[axis] = s.start
                step[axis] = s.step
            elif isinstance(match[axis], tuple):
                t = validate_tuple(match[axis], source.shape[axis])
                shape.append(len(t))
            else:
                elided.append(axis)

        for axis in range(len(match), source.ndim):
            shape.append(source.shape[axis])
            offset[axis] = 0

        super().__init__(shape, source.dtype)
        self._source = source
        self._match = match

        def map_coord(source_coord):
            assert len(source_coord) == source.ndim
            dest_coord = []
            for axis in range(source.ndim):
                if axis in elided:
                    pass
                elif isinstance(source_coord[axis], slice):
                    s = validate_slice(source_coord[axis], source.shape[axis])
                    start = s.start + offset[axis]
                    _step = s.step * step.get(axis, 1)
                    stop = start + (step.get(axis, 1) * (s.stop - s.start))
                    dest_coord.append(slice(start, stop, _step))
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
            assert len(coord) == self.ndim

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
                    at = validate_slice(at, source.shape[axis])
                    start = at.start - offset[axis]
                    _step = at.step * step.get(axis, 1)
                    stop = start + ((s.stop - s.start) // _step)
                    source_coord.append(slice(start, stop, _step))
                elif isinstance(at, tuple):
                    source_coord.append(tuple(c - offset[axis] for c in at))
                else:
                    source_coord.append(at - offset[axis])

            return tuple(source_coord)

        self._map_coord = map_coord
        self._invert_coord = invert_coord

    def __getitem__(self, match):
        match = validate_match(match, self.shape)
        return self._source[source_coord]

    def __setitem__(self, match, value):
        match = validate_match(match, self.shape)
        source_coord = self._invert_coord(match)
        self._source[source_coord] = value

    def filled(self, match = None):
        if match is None:
            for row in self._source.filled(self._match):
                coord = row[:-1]
                value = row[-1]
                yield self._map_coord(coord) + (value,)
        else:
            raise NotImplementedError


def validate_match(match, shape):
    assert len(match) <= len(shape)

    for axis in range(len(match)):
        if match[axis] is None:
            pass
        elif isinstance(match[axis], slice):
            validate_slice(match[axis], shape[axis])
        elif isinstance(match[axis], tuple):
            validate_tuple(match[axis], shape[axis])
        elif match[axis] < 0:
            assert abs(match[axis]) < shape[axis]
        else:
            assert match[axis] < shape[axis]

    return match

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
        stop = dim + stop
    else:
        stop = s.stop

    step = s.step if s.step else 1

    return slice(start, stop, step)


def validate_tuple(t, dim):
    if not t:
        raise IndexError

    if not all(isinstance(match[axis][i], int) for i in range(len(match[axis]))):
        raise IndexError

    if any([abs(match[axis][i]) > dim for i in range(len(match[axis]))]):
        raise IndexError

    return t

