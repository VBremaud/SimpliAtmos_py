import numpy as np
from simpliatmos.model.states import Vector

def copyto(x, y):
    if hasattr(y, "_fields"):
        for k in range(len(x)):
            copyto(x[k], y[k])
    else:
        y[:] = x[:]


def addto_list(y, coefs, x):
    """y += sum_i coefs[i]*x[i]

    - coefs and x are lists
    - x's are either
      - np.array
      - namedtuple of np.array
      - namedtuple of namedtuple of np.array
      - deeper nesting of namedtuple
    - z has to be mutable

    """
    if hasattr(y, "_fields"):
        # y is a nestedtuple and x is a list of nestedtuple

        for k in range(len(x[0])):
            addto_list(y[k], coefs, [z[k] for z in x])
    else:
        assert isinstance(y, np.ndarray)
        # y is a np.array and x is a list of np.array
        y[:] += sum((c*xx for c, xx in zip(coefs, x)))


def addto(y, *args):
    """addto with a more flexible API than addto_list

    instead of giving coefs and x's as list, they are given in an
    alternate sequence of arbitrary length

    addto(y, c0, x0, c1, x1, c2, x2)

    is equivalent to

    addto_list(y, [c0, c1, c2], [x0, x1, x2])

    but
    addto(y, c0, x0)
    addto(y, c0, x0, c1, x1)
    also work

    """
    assert len(args) % 2 == 0
    addto_list(y, args[::2], args[1::2])