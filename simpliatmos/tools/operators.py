from numba import njit, prange, set_num_threads
import numpy as np
import os
from simpliatmos.model.states import Vector
from simpliatmos.tools.weno import compflux_weno3, vortexforce_weno3

# Utiliser tous les cœurs disponibles
set_num_threads(os.cpu_count())

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
        #print(U.x[50,100])
        #print(du.x[50,100])
        #print(omega[50,100])
        #print('test')
        vortexforce_weno3(
            du.y.reshape(-1),
            U.x.reshape(-1),
            omega.reshape(-1),
            mesh.xshift,
            mesh.yshift,
            -1
        )
        #print(U.x[50,100])
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


@njit(parallel=True)
def njit_addgrad(phi, dux, duy, mskx, msky):
    ny, nx = phi.shape
    for j in prange(1, ny):
        for i in range(1, nx):
            dux[j, i] -= (phi[j, i] - phi[j, i-1]) * mskx[j, i]
            duy[j, i] -= (phi[j, i] - phi[j-1, i]) * msky[j, i]


@njit(parallel=True)
def njit_addgrad(phi, dux, duy, mskx, msky):
    ny, nx = phi.shape
    for j in prange(1, ny):
        for i in range(1, nx):
            dux[j, i] -= (phi[j, i] - phi[j, i-1]) * mskx[j, i]
            duy[j, i] -= (phi[j, i] - phi[j-1, i]) * msky[j, i]

def addgrad(mesh, phi, du):
    if isinstance(du, Vector):
        njit_addgrad(phi, du.x, du.y, mesh.mskx, mesh.msky)
    else:
        dummy = np.zeros_like(du)
        njit_addgrad(phi, du, dummy, mesh.mskx, mesh.msky)


def sharp(mesh, u, U):
    if isinstance(u, Vector):
        U.x[:] = u.x * (1 / mesh.dx**2)
        U.y[:] = u.y * (1 / mesh.dy**2)
    else:
        U.x[:] = u * (1 / mesh.dx**2)


def compute_vorticity(mesh, u, omega):
    #print(u.x[50,100])
    #print(u.y[50,100])
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


@njit(parallel=True)
def njit_div(Ux, Uy, delta, msk):
    ny, nx = delta.shape
    for j in prange(ny - 1):
        for i in range(nx - 1):
            delta[j, i] = -(Ux[j, i+1] - Ux[j, i]) - (Uy[j+1, i] - Uy[j, i])
            delta[j, i] *= msk[j, i]

def div(mesh, U, delta):
    njit_div(U.x, U.y, delta, mesh.msk)

def compute_pressure(param, mesh, h, p):
    p[:] = (param.g / mesh.area) * (h + mesh.hb)


def pressure_projection(mesh, U, delta, p, u):
    sharp(mesh, u, U)
    div(mesh, U, delta)
    """
    print("press",u.y[50,100])
    print("pressure",p[:,100])
    print("delta",delta[55,:])
    print(mesh.area)
    """
    mesh.poisson_centers.solve(-delta * mesh.area, p)
    """
    print("press2",u.y[50,100])
    print("pressure",p[:,100])
    
    print("pressure",p[50,100]-p[49,100])
    print("press",p[50,100])
    print("uy",u.y[50,100])
    print(u.y[50,100]-p[50,100]+p[49,100])
    """
    addgrad(mesh, p, u)
    #print("press3",u.y[50,100])
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


def compute_vertical_velocity(mesh, U):
    U.y[1:, :-1] = -np.cumsum(np.diff(U.x[:-1, :], axis=1), axis=0)

def compute_hydrostatic_pressure(mesh, b, p):
    p[:, :] = 0.5*b
    p[-1::-1, :] -= np.cumsum(b[-1::-1, :], axis=0)
    p *= (mesh.msk*mesh.dx**2)