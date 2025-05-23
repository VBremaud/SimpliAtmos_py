import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

class Poisson2D:
    def __init__(self, mesh, location="c", maindiag=0.0):
        assert location == "c", "Seule la location 'c' est supportée ici."
        self.mesh = mesh
        self.location = location
        self.A, self.G = get_Laplacian_sparse(mesh, location, maindiag)
        self.A_LU = splinalg.splu(self.A)

    def solve(self, rhs, sol):
        """
        print("x1", sol[55,:])
        print(np.shape(sol[0,:]))
        print("rhs", rhs[55,:])
        print("G", self.G[55,:])

        from scipy.sparse.linalg import norm
        print("norm:", norm(self.A))
        print(np.sum(self.mesh.msk))             # nb de points "valides"
        print(np.sum(self.G > -1))  # nb de points dans le système linéaire
        print(np.unique(self.G))  
        
        for i in range(45, 56):
            print("sol",i, sol[i,:])
            print("rhs",i, rhs[i,:])
            print("G",i, self.G[i,:])
            print("A",i, self.A[i,:])
        print(test)
        """
        sol[self.G > -1] = self.A_LU.solve(rhs[self.G > -1])
        #print("x2", sol[:,100])
def get_Gindex(msk):
    G = np.zeros(msk.shape, dtype="i")
    G[msk == 0] = -1
    G[msk == 1] = np.arange(np.sum(msk))
    return G


def get_msk_from_mesh(mesh, location):
    if location == "c":
        msk = mesh.msk
    else:
        raise NotImplementedError("Seule la location 'c' est gérée.")

    if not mesh.param.xperiodic:
        return msk

    n = mesh.param.halowidth
    msk_haloed = msk.copy()
    msk_haloed[:, :n] = 0
    msk_haloed[:, -n:] = 0
    return msk_haloed


def get_Laplacian_sparse(mesh, location, maindiag):
    msk = get_msk_from_mesh(mesh, location)
    G = get_Gindex(msk)
    ny, nx = G.shape
    N = np.sum(G > -1)

    data = np.zeros((5 * N,))
    row = np.zeros((5 * N,), dtype="i")
    col = np.zeros((5 * N,), dtype="i")
    counter = 0

    dx2 = mesh.dy / mesh.dx
    dy2 = mesh.dx / mesh.dy

    xperiodic = mesh.param.xperiodic
    yperiodic = False

    n1 = mesh.param.halowidth if xperiodic else 0
    n2 = mesh.param.halowidth if yperiodic else 0

    def add_entry(coef, r, c, counter):
        data[counter] = coef
        row[counter] = r
        col[counter] = c
        return counter + 1

    for j in range(ny):
        for i in range(nx):
            I = G[j, i]
            if I > -1:
                sum_extra_diag = 0

                west = G[j, i - 1] if i > n1 else (G[j, -n1 - 1] if xperiodic else -1)
                east = G[j, i + 1] if i < nx - 1 - n1 else (G[j, n1] if xperiodic else -1)
                south = G[j - 1, i] if j > n2 else -1
                north = G[j + 1, i] if j < ny - 1 - n2 else -1

                if west > -1:
                    counter = add_entry(dx2, I, west, counter)
                    sum_extra_diag += dx2
                if east > -1:
                    counter = add_entry(dx2, I, east, counter)
                    sum_extra_diag += dx2
                if south > -1:
                    counter = add_entry(dy2, I, south, counter)
                    sum_extra_diag += dy2
                if north > -1:
                    counter = add_entry(dy2, I, north, counter)
                    sum_extra_diag += dy2

                counter = add_entry(-sum_extra_diag - maindiag, I, I, counter)

    A = sparse.coo_matrix((data[:counter], (row[:counter], col[:counter])), shape=(N, N))
    return A.tocsc(), G
