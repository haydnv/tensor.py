import itertools
import numpy as np

from btree.table import Index, Schema, Table


class SparseTensorView(object):
    def __init__(self, shape, dtype=np.int32):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)

    def __getitem__(self, _match):
        raise NotImplementedError

    def __setitem__(self, _match, _value):
        raise NotImplementedError

    def filled(self, match=None):
        raise NotImplementedError

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
        value = self.dtype(value)

        if not isinstance(match, tuple):
            coord = (coord,)

        if len(match) > self.ndim:
            raise IndexError

        if not value:
            selector = dict(zip(range(len(match)), match))
            self._table.slice(selector).delete()
            self._table.rebalance()
            return

        affected = []
        for axis in range(len(match)):
            if match[axis] is None:
                affected.append(range(self.shape[axis]))
            elif isinstance(match[axis], slice):
                affected.append(validate_slice(match[axis], self.shape[axis]))
            elif isinstance(match[axis], tuple):
                affected.append(validate_tuple(match[axis], self.shape[axis]))
            elif match[axis] < 0:
                affected.append([self.shape[axis] + match[axis]])
            else:
                affected.append([match[axis]])

        for axis in range(len(match), self.ndim):
            affected.append(range(self.shape[axis]))

        for match in itertools.product(*affected):
            self._table.upsert(match, (value,))

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

        for axis in range(len(match)):
            if match[axis] is None:
                shape.append(source.shape[axis])
            elif isinstance(match[axis], slice):
                s = validate_slice(match[axis], source.shape[axis])
                shape.append((s.stop - s.start) // s.step)
            elif isinstance(match[axis], tuple):
                t = validate_tuple(match[axis], source.shape[axis])
                shape.append(len(t))

        for axis in range(len(match), source.ndim):
            shape.append(source.shape[axis])

        super().__init__(shape, source.dtype)
        self._source = source
        self._match = match

    def filled(self, match = None):
        elide = []
        offset = []

        for axis in range(len(self._match)):
            if isinstance(self._match[axis], int):
                elide.append(axis)
            elif isinstance(self._match[axis], slice):
                offset.append(self._match[axis].start)
            elif isinstance(self._match[axis], tuple):
                offset.append(self._match[axis][0])
            else:
                offset.append(0)

        for axis in range(len(self._match), self._source.ndim):
            offset.append(0)

        def translate(coord):
            assert len(coord) == self._source.ndim
            elided = [coord[i] for i in range(len(coord)) if i not in elide]
            return tuple(elided[i] - offset[i] for i in range(len(elided)))

        if match is None:
            for row in self._source.filled(self._match):
                coord = row[:-1]
                value = row[-1]
                yield translate(coord) + (value,)
        else:
            raise NotImplementedError


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

