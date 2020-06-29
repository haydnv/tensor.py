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

    def __bool__(self, other):
        return self.as_type(np.bool)

    def __eq__(self, other):
        if self.shape != other.shape:
            return False

        if self.dtype == np.bool and other.dtype == np.bool:
            return self ^ other

        return not (self - other)

    def __getitem__(self, _match):
        raise NotImplementedError

    def __setitem__(self, _match, _value):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __xor__(self, other):
        raise NotImplementedError

    def as_type(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def expand_dims(self, axis):
        raise NotImplementedError

    def shuffle(self, _axis):
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


class Permutation(Rebase):
    def __init__(self, source, permutation):
        assert source.ndim == len(permutation)
        super().__init__([source.shape[axis] for axis in permutation])

        self._permute_from = dict(zip(range(self.ndim), permutation))
        self._permute_to = {
            dest_axis: source_axis
            for (source_axis, dest_axis) in self._permute_from.items()}

    def __getitem__(self, match):
        raise NotImplementedError

    def _invert_coord(self, coord):
        if not isinstance(coord, tuple):
            coord = (coord,)

        match = list(coord)
        while len(match) < self.ndim:
            match.append(slice(None))

        return tuple(match[self._permute_from[axis]] for axis in range(self.ndim))

    def _map_coord(self, coord):
        if not isinstance(coord, tuple):
            coord = (coord,)

        match = list(coord)
        while len(match) < self.ndim:
            match.append(slice(None))

        return tuple(match[self._permute_to[axis]] for axis in range(self.ndim))

