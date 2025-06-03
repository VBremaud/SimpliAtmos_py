from numba import njit, prange, set_num_threads
import numpy as np
from simpliatmos.model.states import Vector
from simpliatmos.tools import weno

def addvortexforce(param, mesh, U, omega, du):
    method = param.vortexforce
    if isinstance(du, Vector):
        weno.vortexforce(du.x, U.y, omega, mesh.ov.y,
                         mesh.yshift, mesh.xshift, +1, method)
        weno.vortexforce(du.y, U.x, omega, mesh.ov.x,
                         mesh.xshift, mesh.yshift, -1, method)
   
    else:
        # hydrostatic case
        weno.vortexforce(du, U.y, omega, mesh.ov.y,
                         mesh.yshift, mesh.xshift, +1, method)


def divflux(param, mesh, flx, q, U, dq):
    method = param.compflux
    weno.compflux(flx.x, U.x, q, mesh.oc.x, mesh.xshift,
                  method, param.nthreads)

    weno.compflux(flx.y, U.y, q, mesh.oc.y, mesh.yshift,
                  method, param.nthreads)

    div(mesh, flx, dq)


def addcoriolis(param, mesh, U, du):
    f = param.f0*mesh.area*0.25

    du.x[:-1, 1:-1] += f*(U.y[:-1, :-2]+U.y[1:, :-2] +
                          U.y[:-1, 1:-1]+U.y[1:, 1:-1])

    du.y[1:-1, :-1] -= f*(U.x[:-2, :-1]+U.x[:-2, 1:] +
                          U.x[1:-1, :-1]+U.x[1:-1, 1:])



def addgrad(mesh, phi, du):
    if isinstance(du, Vector):

        du.x[:, 1:] -= np.diff(phi, axis=1)*mesh.mskx[:, 1:]
        du.y[1:, :] -= np.diff(phi, axis=0)*mesh.msky[1:, :]

    else:
        du[:, 1:] -= np.diff(phi, axis=1)*mesh.mskx[:, 1:]


def sharp(mesh, u, U):
    if isinstance(u, Vector):
        U.x[:] = u.x*(1/mesh.dx**2)
        U.y[:] = u.y*(1/mesh.dy**2)
    else:
        U.x[:] = u*(1/mesh.dx**2)


def compute_vorticity(mesh, u, omega):
    if isinstance(u, Vector):

        omega[1:, :] = -np.diff(u.x, axis=0)
        omega[:, 1:] += np.diff(u.y, axis=1)

    else:

        omega[1:, :] = -np.diff(u, axis=0)

    omega *= mesh.slipcoef


def compute_kinetic_energy(param, mesh, u, U, ke):
    method = param.innerproduct

    ke[:] = 0.

    if isinstance(u, Vector):
        if method == "classic":
            ke[:, :-1] = +u.x[:, 1:]*U.x[:, 1:] + u.x[:, :-1]*U.x[:, :-1]
            ke[:-1, :] += u.y[1:, :]*U.y[1:, :] + u.y[:-1, :]*U.y[:-1, :]
            ke *= mesh.msk*0.25
        else:
            weno.innerproduct(ke, U.x, u.x, mesh.ok.x, mesh.xshift, method)
            weno.innerproduct(ke, U.y, u.y, mesh.ok.y, mesh.yshift, method)
            ke *= mesh.msk*0.5

    else:
        if method == "classic":
            ke[:, :-1] = +u[:, 1:]*U.x[:, 1:] + u[:, :-1]*U.x[:, :-1]
            ke *= mesh.msk*0.25
        else:
            weno.innerproduct(ke, U.x, u, mesh.ok.x, mesh.xshift, method)
            ke *= mesh.msk*0.5

def div(mesh, U, delta):
    delta[:, :-1] = -np.diff(U.x, axis=1)
    delta[:-1, :] -= np.diff(U.y, axis=0)
    delta *= mesh.msk


def compute_pressure(param, mesh, h, p):
    p[:] = (param.g/mesh.area) * (h+mesh.hb)


def pressure_projection(mesh, U, delta, p, u):
    sharp(mesh, u, U)
    div(mesh, U, delta)
    mesh.poisson_centers.solve(-delta*mesh.area, p)
    #print("press",p[50,100])
    addgrad(mesh, p, u)
    mesh.fill(u)


def addbuoyancy(mesh, b, du):
    du.y[1:, :] += (0.5*mesh.dy)*(b[1:, :]+b[:-1, :])*mesh.msky[1:, :]


def compute_vertical_velocity(mesh, U):
    U.y[1:, :-1] = -np.cumsum(np.diff(U.x[:-1, :], axis=1), axis=0)

def compute_hydrostatic_pressure(mesh, b, p):
    p[:, :] = 0.5*b
    p[-1::-1, :] -= np.cumsum(b[-1::-1, :], axis=0)
    p *= (mesh.msk*mesh.dx**2)

def fill(mesh, *variables):
    for v in variables:
        mesh.fill(v)