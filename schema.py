import numpy as np


def broadcast(left, right):
    if len(left.shape) < len(right.shape):
        right, left = broadcast(right, left)
        return left, right

    shape = list(left.shape)
    offset = len(shape) - len(right.shape)
    for x in range(len(right.shape)):
        if shape[offset + x] == right.shape[x]:
            pass
        elif right.shape[x] == 1:
            pass
        elif shape[offset + x] == 1:
            shape[offset + x] = right.shape[x]
        else:
            raise ValueError(f"cannot broadcast dimensions {shape[offset + x]} and {right.shape[x]}")

    return left.broadcast(shape), right.broadcast(shape)


def check_permutation(shape, permutation):
    ndim = len(shape)

    if permutation is None:
        permutation = list(reversed(range(ndim)))
    else:
        permutation = [ndim + x if x < 0 else x for x in permutation]

    assert len(permutation) == ndim, f"invalid permutation {permutation} for {ndim} dimensions"
    assert all(0 <= x < ndim for x in permutation)
    assert all(x in permutation for x in range(ndim))

    return permutation


def slice_bounds(source_shape, key):
    assert all(dim > 0 for dim in source_shape)

    bounds = []
    shape = []

    for x, bound in enumerate(key):
        dim = source_shape[x]
        assert dim > 0

        if isinstance(bound, slice):
            start = 0 if bound.start is None else bound.start if bound.start >= 0 else dim + bound.start
            stop = dim if bound.stop is None else bound.stop if bound.stop >= 0 else dim + bound.stop
            step = 1 if bound.step is None else bound.step

            assert stop >= start, f"invalid bounds: {key}"
            assert 0 <= start <= dim, f"bound {bound} has invalid start {start} for dimension {dim}"
            assert 0 < stop <= dim, f"bound {bound} has invalid stop {stop} for dimension {dim}"
            assert 0 < step <= dim, f"bound {bound} has invalid step {step} for dimension {dim}"

            bounds.append(slice(start, stop, step))
            shape.append((stop - start) // step)
        elif isinstance(bound, int):
            i = bound if bound >= 0 else dim + bound
            assert 0 <= i <= dim
            bounds.append(i)
        else:
            raise ValueError(f"invalid bound {bound} (type {type(bound)}) for axis {x}")

    for dim in source_shape[len(key):]:
        bounds.append(slice(0, dim, 1))
        shape.append(dim)

    assert all(dim > 0 for dim in shape)

    return bounds, shape


def strides_for(shape):
    return [0 if dim == 1 else int(np.product(shape[x + 1:])) for x, dim in enumerate(shape)]
