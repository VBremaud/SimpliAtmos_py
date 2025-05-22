from numba import njit, prange, set_num_threads
import os
import numpy as np


# Utiliser tous les cœurs disponibles
set_num_threads(os.cpu_count())


@njit("f8(f8,f8,f8)")
def weno3(qm, q0, qp):
    """
    3-points WENO reconstruction (left-biased)
    """
    eps = 1e-8
    beta1 = (q0 - qm)**2
    beta2 = (qp - q0)**2

    w1 = 1. / (beta1 + eps)**2
    w2 = 2. / (beta2 + eps)**2

    P1 = 1.5*q0 - 0.5*qm
    P2 = 0.5*qp + 0.5*q0

    return (w1 * P1 + w2 * P2) / (w1 + w2)


@njit(parallel=True)
def compflux_weno3(flx, U, q, shift, axis_len):
    """
    Compute flx = U * q_half using WENO3 reconstruction (parallelized)
    """
    for i in prange(2*shift, axis_len - 2*shift):
        qL = weno3(q[i - shift], q[i], q[i + shift])
        flx[i] = U[i] * qL


@njit(parallel=True)
def vortexforce_weno3(du, V, omega, shift, shift2, sign):
    """
    Compute du = sign * (ω reconstructed by WENO3) * V averaged (parallelized)
    """
    N = du.shape[0]
    for i in prange(2*shift, N - 2*shift):
        Vmid = 0.25 * (V[i] + V[i+shift] + V[i-shift2] + V[i+shift-shift2])
        omega_rec = weno3(omega[i - shift], omega[i], omega[i + shift])
        du[i] = sign * omega_rec * Vmid
