import itertools
import numpy as np

from btree.table import Index, Schema, Table


class SparseTensor(object):
    def __init__(self, shape, dtype=np.int32):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.ndim = len(shape)

        self._table = Table(Index(Schema(
            [(i, int) for i in range(self.ndim)],
            [("value", self.dtype)])))
        for i in range(self.ndim):
            self._table.add_index(str(i), [i])

    def __setitem__(self, coord, value):
        value = self.dtype(value)

        if not isinstance(coord, tuple):
            coord = (coord,)

        if len(coord) > self.ndim:
            raise IndexError

        if not value:
            selector = dict(zip(range(len(coord)), coord))
            return self._table.slice(selector).delete()

        affected = []
        for axis in range(len(coord)):
            s = coord[axis]

            if s is None:
                affected.append(range(self.shape[axis]))
            elif isinstance(s, slice):
                if s.start is None:
                    start = 0
                elif s.start < 0:
                    start = self.shape[axis] + s.start
                else:
                    start = s.start

                if s.stop is None:
                    stop = self.shape[axis]
                elif s.stop < 0:
                    stop = self.shape[axis] + stop
                else:
                    stop = s.stop

                step = s.step if s.step else 1
                affected.append(range(start, stop, step))
            else:
                affected.append([s])

        for axis in range(len(coord), self.ndim):
            affected.append(range(self.shape[axis]))

        for coord in itertools.product(*affected):
            self._table.upsert(coord, (value,))

    def to_dense(self):
        dense = np.zeros(self.shape, self.dtype)
        for entry in self._table:
            coord = entry[:-1]
            value = entry[-1]
            dense[coord] = value

        return dense

    def _selection_shape(self, coord):
        if len(coord) > self.ndim:
            raise IndexError

        if len(coord) == self.ndim and all([isinstance(c, int) for c in coord]):
            if any(abs(coord[i]) >= self.shape[i] for i in range(self.ndim)):
                raise IndexError

            return 1

        selection_shape = []
        for i in len(coord):
            if coord[i] is None:
                selection_shape.append(self.shape[i])
            elif isinstance(coord[i], slice):
                if coord[i].start is None:
                    start = 0
                elif coord[i].start > 0:
                    start = coord[i].start
                else:
                    start = self.shape[i] + coord[i].start

                if coord[i].end is None:
                    end = self.shape[i]
                elif coord[i].end > 0:
                    end = coord[i].end
                else:
                    end = self.shape[i] + coord[i].end

                assert end >= start
                selection_shape.append(end - start)

        return selection_shape

