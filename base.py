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

    def __eq__(self, other):
        raise NotImplementedError

    def __getitem__(self, _match):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __setitem__(self, match, value):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __str__(self):
        return str(self.to_nparray())

    def __xor__(self, other):
        raise NotImplementedError

    def all(self):
        raise NotImplementedError

    def any(self):
        raise NotImplementedError

    def as_type(self):
        raise NotImplementedError

    def broadcast(self, shape):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def expand_dims(self, axis):
        raise NotImplementedError

    def product(self, _axis=None):
        raise NotImplementedError

    def sum(self, _axis=None):
        raise NotImplementedError

    def transpose(self, permutation=None):
        raise NotImplementedError

    def to_nparray(self):
        arr = np.zeros(self.shape, self.dtype)
        for coord in itertools.product(*[range(dim) for dim in self.shape]):
            arr[coord] = self[coord]
        return arr



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
    p = 1
    for i in iterable:
        if i:
            p *= i
        else:
            return 0

    return p


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

