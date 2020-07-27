import math

from base import validate_match, validate_slice


class Broadcast(object):
    def __init__(self, source_shape, shape):
        if len(source_shape) > len(shape):
            raise ValueError

        broadcast = [True for _ in range(len(shape))]
        offset = len(shape) - len(source_shape)
        for axis in range(offset, len(shape)):
            if shape[axis] == source_shape[axis - offset]:
                broadcast[axis] = False
            elif shape[axis] == 1 or source_shape[axis - offset] == 1:
                broadcast[axis] = True
            else:
                raise ValueError("cannot broadcast")

        self.shape = tuple(shape)
        self._source_shape = source_shape
        self._broadcast = broadcast
        self._offset = offset

    def invert_coord(self, coord):
        assert len(coord) <= len(self.shape)

        source_coord = []
        for axis in range(len(self._source_shape)):
            if self._broadcast[axis + self._offset]:
                source_coord.append(0)
            else:
                source_coord.append(coord[axis + self._offset])

        return tuple(source_coord)

    def map_coord(self, source_coord):
        assert len(source_coord) == len(self._source_shape)

        coord = [slice(0, dim, 1) for dim in self.shape]
        print("broadcast", self._broadcast, "offset", self._offset)
        for axis in range(len(self._source_shape)):
            if not self._broadcast[axis + self._offset]:
                coord[axis + self._offset] = source_coord[axis]

        return tuple(coord)


class Slice(object):
    def __init__(self, source_shape, match):
        match = validate_match(match, source_shape)

        shape = []
        offset = {}
        elided = []

        for axis in range(len(match)):
            if match[axis] is None:
                shape.append(source_shape[axis])
                offset[axis] = 0
            elif isinstance(match[axis], slice):
                s = match[axis]
                shape.append(math.ceil((s.stop - s.start) / s.step))
                offset[axis] = s.start
            elif isinstance(match[axis], tuple):
                shape.append(len(match[axis]))
            else:
                elided.append(axis)

        for axis in range(len(match), len(source_shape)):
            shape.append(source_shape[axis])
            offset[axis] = 0

        self.match = match
        self.shape = tuple(shape)
        self._source_shape = source_shape
        self._elided = elided
        self._offset = offset

    def map_coord(self, source_coord):
        assert len(source_coord) == len(self._source_shape)
        dest_coord = []
        for axis in range(len(self._source_shape)):
            if axis in self._elided:
                pass
            elif isinstance(source_coord[axis], slice):
                raise NotImplementedError
            elif isinstance(source_coord[axis], tuple):
                dest_coord.append(tuple(c - self._offset[axis] for c in source_coord[axis]))
            else:
                dest_coord.append(source_coord[axis] - self._offset[axis])

        assert len(dest_coord) == len(self.shape)
        return tuple(dest_coord)

    def invert_coord(self, coord):
        coord = [
            coord[i] if i < len(coord) else slice(None)
            for i in range(len(self.shape))]

        source_coord = []
        for axis in range(len(self._source_shape)):
            if axis in self._elided:
                source_coord.append(self.match[axis])
                continue

            at = coord.pop(0)
            if at is None:
                if axis < len(self.match):
                    source_coord.append(self.match[axis])
                else:
                    source_coord.append(None)
            elif isinstance(at, slice):
                if axis < len(self.match):
                    match_axis = self.match[axis]
                else:
                    match_axis = validate_slice(slice(None), self._source_shape[axis])

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
                    elif at.stop < 0:
                        stop = match_axis.stop - (match_axis.step * abs(at.stop))
                    else:
                        at = validate_slice(at, (match_axis.stop - match_axis.start))
                        stop = start + (match_axis.step * (at.stop - at.start))

                    source_coord.append(slice(start, stop, step))
                else:
                    if at.start != 0:
                        raise IndexError

                    source_coord.append(match[axis])
            elif isinstance(at, tuple):
                source_coord.append(tuple(c + self._offset[axis] for c in at))
            else:
                source_coord.append(at + self._offset[axis])

        return tuple(source_coord)

