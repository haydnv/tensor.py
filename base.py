
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

    def random(self):
        raise NotImplementedError

    def shuffle(self, _axis):
        raise NotImplementedError
