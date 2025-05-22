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

def init_state(state, mesh, param):
    x, y = mesh.xy()

    if param.model == "boussinesq":
        # vitesse initiale nulle
        state.u[:, :] = 0.0
        state.U.x[:, :] = 0.0
        state.U.y[:, :] = 0.0

        state.b[:, :] = 10*y #+1e-2*np.random.normal(size=mesh.shape)
        state.b *= mesh.msk

    elif param.model == "euler":
        # Onde sinusoidale en vitesse u
        state.u[:, :] = np.sin(2 * np.pi * x / param.Lx)
        state.U.x[:, :] = state.u[:, :] / mesh.dx**2
        state.U.y[:, :] = 0.0

    elif param.model == "hydrostatic":
        state.b[:, :] = 0.0
        state.uh[:, :] = 0.0
        state.U.x[:, :] = 0.0
        state.U.y[:, :] = 0.0

    else:
        raise ValueError(f"Initialisation non définie pour modèle {param.model}")
