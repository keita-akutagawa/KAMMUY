import numpy as np
import scipy.sparse as sp
import scipy.io as sio


def make_poisson_matrix_symmetric(Nx, Ny, dx, dy):
    Ix = sp.eye(Nx, format='csr')
    Iy = sp.eye(Ny, format='csr')

    ex = np.ones(N) / (2.0 * dx)**2
    Tx = sp.diags([-ex, 2*ex, -ex], [-2, 0, 2], shape=(Nx, Nx), format='lil')

    ey = np.ones(N) / (2.0 * dy)**2
    Ty = sp.diags([-ey, 2*ey, -ey], [-2, 0, 2], shape=(Ny, Ny), format='lil')

    A = sp.kron(Ix, Ty) + sp.kron(Tx, Iy)
    
    return A


Nx, Ny = 5, 300
dx, dy = 10.0, 10.0
N = Nx * Ny

A = make_poisson_matrix_symmetric(Nx, Ny, dx, dy)

sio.mmwrite("poisson_symmetric.mtx", A)


