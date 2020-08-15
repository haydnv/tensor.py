import math

from collections import OrderedDict

from tensor import affected, product, validate_match, validate_slice


class Broadcast(object):
    def __init__(self, source_shape, broadcast_shape):
        if len(source_shape) > len(broadcast_shape):
            print("cannot broadcast {} into {}".format(source_shape, broadcast_shape))
            raise ValueError

        broadcast = [True for _ in range(len(broadcast_shape))]
        offset = len(broadcast_shape) - len(source_shape)
        inverted_axes = []
        for axis in range(offset, len(broadcast_shape)):
            if broadcast_shape[axis] == source_shape[axis - offset]:
                broadcast[axis] = False
                inverted_axes.append(axis)
            elif broadcast_shape[axis] == 1 or source_shape[axis - offset] == 1:
                broadcast[axis] = True
                inverted_axes.append(axis - offset)
            else:
                raise ValueError("cannot broadcast", source_shape, broadcast_shape)

        self.shape = tuple(broadcast_shape)
        self._inverted_axes = inverted_axes
        self._source_shape = source_shape
        self._broadcast = broadcast
        self._offset = offset

    def invert_axes(self, axes):
        return tuple(OrderedDict.fromkeys(self._inverted_axes[x] for x in axes))

    def invert_coord(self, coord):
        assert len(coord) <= len(self.shape)

        source_coord = []
        for axis in range(len(self._source_shape)):
            if axis + self._offset < len(coord):
                if self._broadcast[axis + self._offset]:
                    source_coord.append(0)
                else:
                    if axis + self._offset < len(coord):
                        source_coord.append(coord[axis + self._offset])
            else:
                source_coord.append(slice(0, self._source_shape[axis]))

        return tuple(source_coord)

    def map_coord(self, source_coord):
        assert len(source_coord) == len(self._source_shape)

        coord = [slice(0, dim, 1) for dim in self.shape]
        for axis in range(len(self._source_shape)):
            if not self._broadcast[axis + self._offset]:
                coord[axis + self._offset] = source_coord[axis]

        return tuple(coord)

class Expand(object):
    def __init__(self, shape, axis):
        if axis > len(shape):
            raise ValueError

        self._source_shape = shape

        inverted_axes = list(range(len(shape)))
        inverted_axes.insert(axis, axis)

        shape = list(shape)
        shape.insert(axis, 1)

        self.shape = tuple(shape)
        self._expand = axis

    def invert_axes(self, axes):
        return tuple(OrderedDict.fromkeys(self._inverted_axes[x] for x in axes))

    def invert_coord(self, coord):
        validate_match(coord, self.shape)

        if len(coord) < self._expand:
            return coord
        else:
            coord = list(coord)
            del coord[self._expand]
            return tuple(coord)

    def map_coord(self, source_coord):
        validate_match(source_coord, self._source_shape)

        if len(source_coord) < self._expand:
            return source_coord
        else:
            coord = list(source_coord)
            coord.insert(self._expand, 0)
            return tuple(coord)


class Slice(object):
    def __init__(self, source_shape, match):
        match = validate_match(match, source_shape)

        shape = []
        offset = {}
        elided = []
        inverted_axes = []

        for axis in range(len(match)):
            if match[axis] is None:
                shape.append(source_shape[axis])
                inverted_axes.append(axis)
                offset[axis] = 0
            elif isinstance(match[axis], slice):
                s = match[axis]
                shape.append(math.ceil((s.stop - s.start) / s.step))
                inverted_axes.append(axis)
                offset[axis] = s.start
            elif isinstance(match[axis], tuple):
                shape.append(len(match[axis]))
                inverted_axes.append(axis)
            else:
                elided.append(axis)

        for axis in range(len(match), len(source_shape)):
            shape.append(source_shape[axis])
            inverted_axes.append(axis)
            offset[axis] = 0

        self.match = match
        self.shape = tuple(shape)
        self._inverted_axes = inverted_axes
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

    def invert_axes(self, axes):
        assert len(axes) <= len(self.shape)
        assert all(x < len(self.shape) for x in axes)
        return tuple(self._inverted_axes[x] for x in axes)

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


class Transpose(object):
    def __init__(self, shape, permutation=None):
        if not permutation:
            permutation = list(reversed(list(axis for axis in range(len(shape)))))

        assert len(permutation) == len(shape)
        assert all(permutation[axis] < len(shape) for axis in range(len(permutation)))

        self.permutation = permutation
        self.shape = tuple(shape[permutation[axis]] for axis in range(len(permutation)))

    def invert_axes(self, axes):
        return tuple(self.permutation[x] for x in axes)

    def invert_coord(self, coord):
        if not isinstance(coord, tuple):
            coord = (coord,)

        source_coord = [slice(None)] * len(self.shape)
        for axis in range(len(coord)):
            source_coord[self.permutation[axis]] = coord[axis]

        return tuple(source_coord)

    def map_coord(self, coord):
        assert len(coord) == len(self.shape)

        if not isinstance(coord, tuple):
            coord = (coord,)

        return tuple(coord[self.permutation[axis]] for axis in range(len(coord)))

    def map_coord_axes(self, partial_source_coord, axes):
        source_coord = {axis: None for axis in range(len(self.shape))}
        for (axis, i) in zip(axes, partial_source_coord):
            source_coord[axis] = i

        coord = []
        for i in range(len(self.permutation)):
            if source_coord[self.permutation[i]] is not None:
                coord.append(source_coord[self.permutation[i]])

        return tuple(coord)

