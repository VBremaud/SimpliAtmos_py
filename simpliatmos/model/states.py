import numpy as np
from collections import namedtuple

Vector = namedtuple("Vector", ("x", "y"))

class State:
    def __init__(self, param, shape):
        model = param.model

        if model == "boussinesq":
            self.u = np.zeros(shape)
            self.b = np.zeros(shape)
            self.U = Vector(np.zeros(shape), np.zeros(shape))
            self.omega = np.zeros(shape)
            self.ke = np.zeros(shape)
            self.p = np.zeros(shape)
            self.div = np.zeros(shape)
            self.flx = Vector(np.zeros(shape), np.zeros(shape))

        elif model == "euler":
            self.u = np.zeros(shape)
            self.U = Vector(np.zeros(shape), np.zeros(shape))
            self.omega = np.zeros(shape)
            self.ke = np.zeros(shape)
            self.p = np.zeros(shape)
            self.div = np.zeros(shape)
            self.flx = Vector(np.zeros(shape), np.zeros(shape))

        elif model == "hydrostatic":
            self.uh = np.zeros(shape)
            self.b = np.zeros(shape)
            self.U = Vector(np.zeros(shape), np.zeros(shape))
            self.omega = np.zeros(shape)
            self.ke = np.zeros(shape)
            self.p = np.zeros(shape)
            self.div = np.zeros(shape)
            self.flx = Vector(np.zeros(shape), np.zeros(shape))

        else:
            raise ValueError(f"Model {model} not supported in this version.")


class Prognostic:
    def __init__(self, param, shape):
        model = param.model

        if model == "boussinesq":
            self.u = np.zeros(shape)
            self.b = np.zeros(shape)

        elif model == "euler":
            self.u = np.zeros(shape)

        elif model == "hydrostatic":
            self.uh = np.zeros(shape)
            self.b = np.zeros(shape)

        else:
            raise ValueError(f"Model {model} not supported in this version.")
