import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags

class Poisson2D:
    def __init__(self, mesh):
        self.mesh = mesh
        self.ny, self.nx = mesh.shape
        self.N = self.nx * self.ny

        # Matrice de Laplacien 2D simplifiée (Dirichlet)
        dx2 = mesh.dx**2
        dy2 = mesh.dy**2
        diag = -2.0 / dx2 - 2.0 / dy2
        offx = 1.0 / dx2
        offy = 1.0 / dy2

        main = diag * np.ones(self.N)
        east_west = offx * np.ones(self.N - 1)
        north_south = offy * np.ones(self.N - mesh.nx)

        for i in range(1, self.ny):
            east_west[i * self.nx - 1] = 0  # couper les sauts de ligne

        self.A = diags(
            [main, east_west, east_west, north_south, north_south],
            [0, -1, 1, -mesh.nx, mesh.nx],
            shape=(self.N, self.N),
            format="csr"
        )

    def solve(self, rhs, sol):
        # Flatten + solve + reshape
        b = rhs.reshape(-1)
        x, info = cg(self.A, b)
        sol[:] = x.reshape(self.mesh.shape)
