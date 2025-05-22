import numpy as np
from simpliatmos.core.poisson import Poisson2D

class Mesh:
    def __init__(self, param):
        self.nx = param.nx
        self.ny = param.ny
        self.dx = param.Lx / self.nx
        self.dy = param.Ly / self.ny
        self.area = self.dx * self.dy
        self.shape = (self.ny + 2 * param.halowidth, self.nx + 2 * param.halowidth)
        
        self.xshift = 1
        self.yshift = self.shape[1]  
        self.param = param
        self.slipcoef = np.ones(self.shape)
        self.set_default_mask()
        self.set_masks()
        self.poisson_centers = Poisson2D(self)
    

    def set_default_mask(self):
        nh = self.param.halowidth
        self.msk = np.zeros(self.shape, dtype="i1")
        self.msk[nh:-nh, nh:-nh] = 1

    def set_masks(self):
        self.mskx = np.zeros_like(self.msk, dtype="i1")
        self.mskx[:, 1:] = self.msk[:, 1:] * self.msk[:, :-1]

        self.msky = np.zeros_like(self.msk, dtype="i1")
        self.msky[1:, :] = self.msk[1:, :] * self.msk[:-1, :]

    def x(self, which):
        idx = get_idx(self.param, self.param.xperiodic, self.nx)
        shift = 0.5 if which in ["c", "y"] else 0.0
        return (idx+shift)*self.dx

    def y(self, which):
        idx = get_idx(self.param, self.param.yperiodic, self.ny)
        shift = 0.5 if which in ["c", "x"] else 0.0
        return (idx+shift)*self.dy

    def xy(self, which="c"):
        return np.meshgrid(self.x(which), self.y(which))

    def fill(self, variable):
        """Applique le remplissage des bords pour la périodicité en x"""
        if self.param.xperiodic:
            n = self.param.halowidth
            if isinstance(variable, np.ndarray):
                variable[:, :n] = variable[:, -2*n:-n]
                variable[:, -n:] = variable[:, n:2*n]
            elif hasattr(variable, "_fields"):  # cas d’un Vector
                for field in variable:
                    self.fill(getattr(variable, field))
        # tu pourras étendre pour yperiodic, ou autres BCs
    
def get_idx(param, periodic, n):
    return np.arange(n+2*param.halowidth)-param.halowidth