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

    def __getitem__(self, _match):
        raise NotImplementedError

    def __setitem__(self, _match, _value):
        raise NotImplementedError

    def __eq__(self, other):
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


class SparseTensorView(Tensor):
    def __init__(self, shape, dtype, default):
        super().__init__(shape, dtype)
        self._default = default

    def to_dense(self):
        dense = np.ones(self.shape, self.dtype) * self._default
        for entry in self.filled():
            coord = entry[:-1]
            value = entry[-1]
            dense[coord] = value

        return dense


class Permutation(Tensor):
    def __init__(self, source, permutation):
        assert source.ndim == len(permutation)
        super().__init__(
            [source.shape[axis] for axis in range(source.ndim)], source.dtype)

        self._source = source
        self._permute_from = dict(zip(range(self.ndim), permutation))
        self._permute_to = {
            dest_axis: source_axis
            for (source_axis, dest_axis) in self._permute_from.items()}

    def __getitem__(self, match):
        to_permute = self._source[self._invert_coord(match)]
        if isinstance(to_permute, self.dtype):
            return to_permute
        else:
            permutation = [self._permute_from[axis] for axis in range(self.ndim)]
            return Permutation(to_permute, permutation)

    def __setitem__(self, match, value):
        self._source[self._invert_coord(match)] = value

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

