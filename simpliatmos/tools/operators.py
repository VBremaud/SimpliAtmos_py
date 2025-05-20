import numpy as np
from simpliatmos.model.states import Vector
from simpliatmos.tools.weno import compflux_weno3, vortexforce_weno3


def addvortexforce(param, mesh, U, omega, du):
    if isinstance(du, Vector):
        vortexforce_weno3(
            du.x.reshape(-1),
            U.y.reshape(-1),
            omega.reshape(-1),
            mesh.yshift,
            mesh.xshift,
            +1
        )
        vortexforce_weno3(
            du.y.reshape(-1),
            U.x.reshape(-1),
            omega.reshape(-1),
            mesh.xshift,
            mesh.yshift,
            -1
        )
    else:
        vortexforce_weno3(
            du.reshape(-1),
            U.y.reshape(-1),
            omega.reshape(-1),
            mesh.yshift,
            mesh.xshift,
            +1
        )


def divflux(param, mesh, flx, q, U, dq):
    compflux_weno3(
        flx.x.reshape(-1),
        U.x.reshape(-1),
        q.reshape(-1),
        mesh.xshift,
        q.size
    )
    compflux_weno3(
        flx.y.reshape(-1),
        U.y.reshape(-1),
        q.reshape(-1),
        mesh.yshift,
        q.size
    )
    div(mesh, flx, dq)


def addcoriolis(param, mesh, U, du):
    f = param.f0 * mesh.area * 0.25
    du.x[:-1, 1:-1] += f * (U.y[:-1, :-2] + U.y[1:, :-2] +
                            U.y[:-1, 1:-1] + U.y[1:, 1:-1])
    du.y[1:-1, :-1] -= f * (U.x[:-2, :-1] + U.x[:-2, 1:] +
                            U.x[1:-1, :-1] + U.x[1:-1, 1:])


def addgrad(mesh, phi, du):
    if isinstance(du, Vector):
        du.x[:, 1:] -= np.diff(phi, axis=1) * mesh.mskx[:, 1:]
        du.y[1:, :] -= np.diff(phi, axis=0) * mesh.msky[1:, :]
    else:
        du[:, 1:] -= np.diff(phi, axis=1) * mesh.mskx[:, 1:]


def sharp(mesh, u, U):
    if isinstance(u, Vector):
        U.x[:] = u.x * (1 / mesh.dx**2)
        U.y[:] = u.y * (1 / mesh.dy**2)
    else:
        U.x[:] = u * (1 / mesh.dx**2)


def compute_vorticity(mesh, u, omega):
    if isinstance(u, Vector):
        omega[1:, :] = -np.diff(u.x, axis=0)
        omega[:, 1:] += np.diff(u.y, axis=1)
    else:
        omega[1:, :] = -np.diff(u, axis=0)

    omega *= mesh.slipcoef


def compute_kinetic_energy(param, mesh, u, U, ke):
    ke[:] = 0.0
    if isinstance(u, Vector):
        ke[:, :-1] = u.x[:, 1:] * U.x[:, 1:] + u.x[:, :-1] * U.x[:, :-1]
        ke[:-1, :] += u.y[1:, :] * U.y[1:, :] + u.y[:-1, :] * U.y[:-1, :]
        ke *= mesh.msk * 0.25
    else:
        ke[:, :-1] = u[:, 1:] * U.x[:, 1:] + u[:, :-1] * U.x[:, :-1]
        ke *= mesh.msk * 0.25


def div(mesh, U, delta):
    delta[:, :-1] = -np.diff(U.x, axis=1)
    delta[:-1, :] -= np.diff(U.y, axis=0)
    delta *= mesh.msk


def compute_pressure(param, mesh, h, p):
    p[:] = (param.g / mesh.area) * (h + mesh.hb)


def pressure_projection(mesh, U, delta, p, u):
    sharp(mesh, u, U)
    div(mesh, U, delta)
    mesh.poisson_centers.solve(-delta * mesh.area, p)
    addgrad(mesh, p, u)
    mesh.fill(u)


def addbuoyancy(mesh, b, du):
    if isinstance(du, Vector):
        du.y[1:, :] += 0.5 * mesh.dy * (b[1:, :] + b[:-1, :]) * mesh.msky[1:, :]
    else:
        du[1:, :] += 0.5 * mesh.dy * (b[1:, :] + b[:-1, :]) * mesh.msky[1:, :]


def fill(mesh, *variables):
    for var in variables:
        if hasattr(var, "_fields"):  # c’est un vecteur (namedtuple)
            for v in var:
                fill(mesh, v)
        else:
            pass  # pas encore de halo à remplir
