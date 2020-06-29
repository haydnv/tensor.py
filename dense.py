import itertools

from base import Tensor


class BlockTensor(Tensor):
    def __init__(self, source):
        super().__init__(source.shape)
        self._source = source

    def filled(self):
        for coord in itertools.product(*self._source.shape):
            value = source[coord]
            if value != 0:
                yield tuple(coord) + (value,)
