import itertools

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

    def filled(self):
        raise NotImplementedError


class BlockTensor(Tensor):
    def __init__(self, source):
        super().__init__(source.shape)
        self._source = source

    def filled(self):
        for coord in itertools.product(*self._source.shape):
            value = source[coord]
            if value != 0:
                yield tuple(coord) + (value,)

