import functools
import itertools
import math
import numpy as np

from collections import OrderedDict


class Tensor(object):
    def __init__(self, shape, dtype=np.int32):
        if not all(isinstance(dim, int) for dim in shape):
            raise ValueError

        if any(dim < 0 for dim in shape):
            raise ValueError

        self.dtype = dtype
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.size = product(shape)

    def __eq__(self, _other):
        raise NotImplementedError

    def __getitem__(self, _match):
        raise NotImplementedError

    def __mul__(self, other):
        if other.ndim > self.ndim:
            return other * self

        raise NotImplementedError

    def __setitem__(self, match, value):
        dest = self[match]
        value = value.broadcast(dest.shape)

        for coord in itertools.product(*[range(dim) for dim in dest.shape]):
            dest[coord] = value[coord]

    def __sub__(self, other):
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

        subtraction = Tensor(left.shape)
        for coord in itertools.product(*[range(dim) for dim in left.shape]):
            subtraction[coord] = left[coord] - right[coord]

        return subtraction

    def __str__(self):
        return str(self.to_nparray())

    def __xor__(self, other):
        this = bool(self)
        that = bool(other)

        if that.ndim > this.ndim:
            return that ^ this

        if this.shape != that.shape:
            that = that.broadcast(this.shape)

        xor = Tensor(this.shape, np.bool)
        for coord in itertools.product(*[range(dim) for dim in this.shape]):
            xor[coord] = this[coord] ^ that[coord]

        return xor

    def all(self):
        for coord in itertools.product(*[range(dim) for dim in self.shape]):
            if not self[coord]:
                return False

        return True

    def any(self):
        for coord in itertools.product(*[range(dim) for dim in self.shape]):
            if self[coord]:
                return True

        return False

    def broadcast(self, shape):
        if shape == self.shape:
            return self

        return Broadcast(self, shape)

    def as_type(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def expand_dims(self, axis):
        return Expansion(self, axis)

    def product(self, _axis):
        raise NotImplementedError

    def shuffle(self, _axis):
        raise NotImplementedError

    def sum(self, _axis):
        raise NotImplementedError

    def to_nparray(self):
        arr = np.zeros(self.shape, self.dtype)
        for coord in itertools.product(*[range(dim) for dim in self.shape]):
            arr[coord] = self[coord]
        return arr

    def transpose(self, permutation=None):
        return Permutation(self, permutation)


class Rebase(Tensor):
    def __init__(self, source, shape):
        Tensor.__init__(self, shape, source.dtype)
        self._source = source

    def __getitem__(self, coord):
        if not isinstance(coord, tuple):
            coord = (coord,)

        return self._source[self._invert_coord(coord)]

    def __setitem__(self, coord, value):
        if not isinstance(coord, tuple):
            coord = (coord,)

        self._source[self._invert_coord(coord)] = value

    def _invert_coord(self, coord):
        raise NotImplementedError

    def _map_coord(self, coord):
        raise NotImplementedError


class Broadcast(Rebase):
    def __init__(self, source, shape):
        if source.ndim > len(shape):
            raise ValueError

        Rebase.__init__(self, source, shape)

        broadcast = [True for _ in range(self.ndim)]
        offset = self.ndim - source.ndim
        for axis in range(offset, self.ndim):
            if self.shape[axis] == source.shape[axis - offset]:
                broadcast[axis] = False
            elif self.shape[axis] == 1 or source.shape[axis - offset] == 1:
                broadcast[axis] = True
            else:
                raise ValueError("cannot broadcast")

        self._broadcast = broadcast
        self._offset = offset

    def __setitem__(self, _coord, _value):
        raise IndexError

    def _invert_coord(self, coord):
        assert len(coord) <= self.ndim

        source_coord = []
        for axis in range(self._source.ndim):
            if self._broadcast[axis + self._offset]:
                source_coord.append(0)
            else:
                source_coord.append(coord[axis + self._offset])

        return tuple(source_coord)

    def _map_coord(self, source_coord):
        coord = [slice(None) for _ in range(self.ndim)]
        for axis in range(self._source.ndim):
            if not self._broadcast[axis + self._offset]:
                coord[axis + self._offset] = source_coord[axis]

        return tuple(coord)


class Expansion(Rebase):
    def __init__(self, source, axis):
        if axis > source.ndim:
            raise ValueError

        shape = list(source.shape)
        shape.insert(axis, 1)
        Rebase.__init__(self, source, shape)

        self._expand = axis

    def _invert_coord(self, coord):
        validate_match(coord, self.shape)

        if len(coord) < self._expand:
            return coord
        else:
            coord = list(coord)
            del coord[self._expand]
            return tuple(coord)

    def _map_coord(self, source_coord):
        validate_match(source_coord, self._source.shape)

        if len(source_coord) < self._expand:
            return source_coord
        else:
            coord = list(source_coord)
            coord.insert(self._expand, 0)
            return tuple(coord)


class Permutation(Rebase):
    def __init__(self, source, permutation=None):
        if not permutation:
            permutation = list(reversed(list(axis for axis in range(source.ndim))))

        assert len(permutation) == source.ndim
        assert all(permutation[axis] < source.ndim for axis in range(len(permutation)))

        self._permutation = permutation

        shape = [source.shape[permutation[axis]] for axis in range(source.ndim)]
        Rebase.__init__(self, source, shape)

    def __getitem__(self, coord):
        source = self._source[self._invert_coord(coord)]
        if source.shape == tuple():
            return source

        permutation = OrderedDict(zip(range(self.ndim), self._permutation))
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

    def _invert_coord(self, coord):
        if not isinstance(coord, tuple):
            coord = (coord,)

        source_coord = [slice(None)] * self.ndim
        for axis in range(len(coord)):
            source_coord[self._permutation[axis]] = coord[axis]

        return tuple(source_coord)

    def _map_coord(self, coord):
        if not isinstance(coord, tuple):
            coord = (coord,)

        return tuple(coord[self._permutation[axis]] for axis in range(len(coord)))


class TensorSlice(Rebase):
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

        Rebase.__init__(self, source, tuple(shape))
        self._source = source
        self._match = match
        self._elided = elided
        self._offset = offset

    def _map_coord(self, source_coord):
        assert len(source_coord) == self._source.ndim
        dest_coord = []
        for axis in range(self._source.ndim):
            if axis in self._elided:
                pass
            elif isinstance(source_coord[axis], slice):
                raise NotImplementedError
            elif isinstance(source_coord[axis], tuple):
                dest_coord.append(tuple(c - self._offset[axis] for c in source_coord[axis]))
            else:
                dest_coord.append(source_coord[axis] - self._offset[axis])

        assert len(dest_coord) == self.ndim
        return tuple(dest_coord)

    def _invert_coord(self, coord):
        coord = [
            coord[i] if i < len(coord) else slice(None)
            for i in range(self.ndim)]

        source_coord = []
        for axis in range(self._source.ndim):
            if axis in self._elided:
                source_coord.append(self._match[axis])
                continue

            at = coord.pop(0)
            if at is None:
                if axis < len(self._match):
                    source_coord.append(self._match[axis])
                else:
                    source_coord.append(None)
            elif isinstance(at, slice):
                if axis < len(self._match):
                    match_axis = self._match[axis]
                else:
                    match_axis = validate_slice(slice(None), self._source.shape[axis])

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


def affected(match, shape):
    affected = []
    for axis in range(len(match)):
        if match[axis] is None:
            affected.append(range(shape[axis]))
        elif isinstance(match[axis], slice):
            s = match[axis]
            affected.append(range(s.start, s.stop, s.step))
        elif isinstance(match[axis], tuple):
            affected.append(match[axis])
        elif match[axis] < 0:
            affected.append([shape[axis] + match[axis]])
        else:
            affected.append([match[axis]])

    for axis in range(len(match), len(shape)):
        affected.append(range(shape[axis]))

    return affected

def product(iterable):
    return functools.reduce(lambda p, i: p * i, iterable, 1)

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
            match[axis] = shape[axis] + match[axis]
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

    start = min(start, dim)

    if s.stop is None:
        stop = dim
    elif s.stop < 0:
        stop = dim + s.stop
    else:
        stop = s.stop

    stop = min(stop, dim)

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

