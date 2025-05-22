import numpy as np

def init_state(state, mesh, param):
    x, y = mesh.xy()

    if param.model == "boussinesq":
        # vitesse initiale nulle
        state.u.x[:, :] = 0.0
        state.u.y[:, :] = 0.0
        state.U.x[:, :] = 0.0
        state.U.y[:, :] = 0.0

        b = state.b
        b[:, :] = 10*y +1e-2*np.random.normal(size=mesh.shape)
        b *= mesh.msk
        b = state.b

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
