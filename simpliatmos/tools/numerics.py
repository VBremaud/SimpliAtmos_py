import numpy as np
from simpliatmos.model.states import Vector

def copyto(x, y):
    if hasattr(y, "_fields"):
        for k in range(len(x)):
            copyto(x[k], y[k])
    else:
        y[:] = x[:]

def addto_list(y, coefs, xlist):
    if isinstance(y, np.ndarray):
        y[:] += sum(c * x for c, x in zip(coefs, xlist))

    elif isinstance(y, Vector):  # ton namedtuple Vector
        addto_list(y.x, coefs, [x.x for x in xlist])
        addto_list(y.y, coefs, [x.y for x in xlist])

    elif hasattr(y, "__dict__"):
        # on filtre uniquement les attributs présents dans tous les x
        common_attrs = [
            attr for attr in y.__dict__
            if all(hasattr(x, attr) for x in xlist)
        ]
        for attr in common_attrs:
            addto_list(getattr(y, attr), coefs, [getattr(x, attr) for x in xlist])

    else:
        raise TypeError(f"Unsupported type in addto_list: {type(y)}")



def addto(y, *args):
    assert len(args) % 2 == 0
    addto_list(y, args[::2], args[1::2])
