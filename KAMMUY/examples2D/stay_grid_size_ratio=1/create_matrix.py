import numpy as np
import scipy.sparse as sp
import scipy.io as sio


def make_poisson_matrix_periodic(Nx, Ny, dx, dy):
    Ix = sp.eye(Nx, format='csr')
    Iy = sp.eye(Ny, format='csr')

    ex = np.ones(N) / (2.0 * dx)**2
    Tx = sp.diags([-ex, 2*ex, -ex], [-2, 0, 2], shape=(Nx, Nx), format='lil')
    Tx[0, -2] = -1 / (2.0 * dx)**2 
    Tx[1, -1] = -1 / (2.0 * dx)**2 
    Tx[-2, 0] = -1 / (2.0 * dx)**2 
    Tx[-1, 1] = -1 / (2.0 * dx)**2 

    ey = np.ones(N) / (2.0 * dy)**2
    Ty = sp.diags([-ey, 2*ey, -ey], [-2, 0, 2], shape=(Ny, Ny), format='lil')
    Ty[0, -2] = -1 / (2.0 * dy)**2 
    Ty[1, -1] = -1 / (2.0 * dy)**2 
    Ty[-2, 0] = -1 / (2.0 * dy)**2 
    Ty[-1, 1] = -1 / (2.0 * dy)**2 

    A = sp.kron(Ix, Ty) + sp.kron(Tx, Iy)
    
    return A


Nx, Ny = 400, 400
dx, dy = 1.0, 1.0
N = Nx * Ny

A = make_poisson_matrix_periodic(Nx, Ny, dx, dy)

sio.mmwrite("poisson_periodic.mtx", A)


