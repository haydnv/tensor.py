import itertools
import numpy as np

class Tensor(object):
    def __init__(self, shape, dtype=np.int32):
        if not all(isinstance(dim, int) for dim in shape):
            raise ValueError

        if any(dim < 0 for dim in shape):
            raise ValueError

        self.dtype = dtype
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def __eq__(self, other):
        if self.shape != other.shape:
            return False

        if self.dtype == np.bool and other.dtype == np.bool:
            return self ^ other

        return ~ (self - other)

    def __getitem__(self, _match):
        raise NotImplementedError

    def __setitem__(self, _match, _value):
        raise NotImplementedError

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
        return "{}".format(self.to_dense())

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

    def as_type(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def expand_dims(self, axis):
        return Expanded(self, axis)

    def product(self, _axis):
        raise NotImplementedError

    def shuffle(self, _axis):
        raise NotImplementedError

    def sum(self, _axis):
        raise NotImplementedError

    def transpose(self, permutation=None):
        if not permutation:
            permutation = list(reversed(list(axis for axis in range(self.ndim))))

        return Permutation(self, permutation)


class Rebase(Tensor):
    def __init__(self, source, shape):
        Tensor.__init__(self, shape, source.dtype)
        self._source = source

    def __getitem__(self, coord):
        return self._source[self._invert_coord(coord)]

    def __setitem__(self, coord, value):
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
            elif self.shape[axis] != 1 and source.shape[axis - offset] == 1:
                broadcast[axis] = True
            else:
                print(self.shape, source.shape, axis, offset)
                raise ValueError("cannot broadcast")

        self._broadcast = broadcast
        self._offset = offset

    def __setitem__(self, _coord, _value):
        raise IndexError

    def _invert_coord(self, coord):
        raise NotImplementedError

    def _map_coord(self, source_coord):
        coord = [slice(None) for _ in range(self.ndim)]
        for axis in range(self._source.ndim):
            if not self._broadcast[axis + self._offset]:
                coord[axis + self._offset] = source_coord[axis]

        return tuple(coord)


class Expanded(Rebase):
    def __init__(self, source, axis):
        if axis >= source.ndim:
            raise ValueError

        shape = list(source.shape)
        shape.insert(axis, 1)
        Rebase.__init__(self, source, shape)

        self._expand = axis

    def _invert_coord(self, coord):
        validate_match(coord, self.shape)

        if len(coord) < self._expand:
            return match
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

