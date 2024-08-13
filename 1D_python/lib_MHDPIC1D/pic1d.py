import numpy as np


def get_rho(q, x, n_x, dx, rho):
    x_index = np.floor(x[0, :] / dx).astype(np.int64)

    cx1 = (x[0, :] - x_index * dx)/dx  
    cx2 = ((x_index + 1) * dx - x[0, :])/dx 
    index_one_array = x_index

    rho += np.bincount(index_one_array, 
                       weights=q * cx2,
                       minlength=n_x
                      )
    rho += np.roll(np.bincount(index_one_array, 
                               weights=q * cx1,
                               minlength=n_x
                              ), 1, axis=0)
    
    return rho


def get_current_density(c, q, v, x, n_x, dx, current):
    x_index = np.floor(x[0, :] / dx).astype(int)
    x_index_half = np.floor((x[0, :] - 0.5 * dx) / dx).astype(int)
    x_index_half_minus = np.where(x_index_half == -1)
    x_index_half[x_index_half == -1] = n_x-1

    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0) / c)**2)

    cx1 = (x[0, :] - (x_index_half + 0.5) * dx)/dx  
    cx2 = ((x_index_half + 1.5) * dx - x[0, :])/dx 
    cx1[x_index_half_minus] = (x[0, x_index_half_minus] - (-0.5) * dx)/dx  
    cx2[x_index_half_minus] = ((0.5)*dx - x[0, x_index_half_minus])/dx 
    index_one_array = x_index_half

    current[0, :] += np.bincount(index_one_array, 
                                 weights=q * v[0, :] / gamma * cx2, 
                                 minlength=n_x
                                )
    current[0, :] += np.roll(np.bincount(index_one_array, 
                                         weights=q * v[0, :] / gamma * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)

    cx1 = (x[0, :] - x_index * dx) / dx  
    cx2 = ((x_index + 1) * dx - x[0, :]) / dx 
    index_one_array = x_index
    
    current[1, :] += np.bincount(index_one_array, 
                                    weights=q * v[1, :] / gamma * cx2, 
                                    minlength=n_x
                                    )
    current[1, :] += np.roll(np.bincount(index_one_array, 
                                            weights=q * v[1, :] / gamma * cx1, 
                                            minlength=n_x
                                            ), 1, axis=0)

    current[2, :] += np.bincount(index_one_array, 
                                 weights=q * v[2, :] / gamma * cx2, 
                                 minlength=n_x
                                )
    current[2, :] += np.roll(np.bincount(index_one_array, 
                                         weights=q * v[2, :] / gamma * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    
    return current


def buneman_boris_v(c, dt, q, m, E, B, v):

    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0) / c)**2)

    #TとSの設定
    T = (q/m) * dt * B / 2.0 / gamma
    S = 2.0 * T / (1.0 + np.linalg.norm(T, axis=0)**2)

    #時間発展
    v_minus = v + (q/m) * E * (dt/2)
    v_0 = v_minus + np.cross(v_minus, T, axis=0)
    v_plus = v_minus + np.cross(v_0, S, axis=0)
    v = v_plus + (q/m) * E * (dt/2.0)

    return v 


def buneman_boris_x(c, dt, v, x):

    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0) / c)**2)

    x = x + v * dt / gamma

    return x


def time_evolution_v(c, E, B, x, q, m, n_x, dx, dt, v):
    E_tmp = E.copy()
    B_tmp = B.copy()
    E_tmp[0, :] = (E[0, :] + np.roll(E[0, :], 1, axis=0)) / 2.0
    B_tmp[0, :] = (B[0, :] + np.roll(B[0, :], -1, axis=0)) / 2.0
    x_index = np.floor(x[0, :] / dx).astype(int)
    x_index_half = np.floor((x[0, :] - 1/2*dx) / dx).astype(int)
    x_index_half_minus = np.where(x_index_half == -1)
    x_index_half[x_index_half == -1] = n_x-1
    E_particle = np.zeros(x.shape)
    B_particle = np.zeros(x.shape)

    #電場
    cx1 = (x[0, :] - x_index*dx)/dx  
    cx2 = ((x_index+1)*dx - x[0, :])/dx 
    cx1 = cx1.reshape(-1, 1)
    cx2 = cx2.reshape(-1, 1)
    E_particle[:, :] = (E_tmp[:, x_index].T * cx2 \
                     + E_tmp[:, (x_index+1)%n_x].T * cx1 \
                    ).T
    
    #磁場
    cx1 = (x[0, :] - (x_index_half + 1/2)*dx)/dx  
    cx2 = ((x_index_half + 3/2)*dx - x[0, :])/dx 
    cx1[x_index_half_minus] = (x[0, x_index_half_minus] - (-1/2)*dx)/dx  
    cx2[x_index_half_minus] = ((1/2)*dx - x[0, x_index_half_minus])/dx 
    cx1 = cx1.reshape(-1, 1)
    cx2 = cx2.reshape(-1, 1)
    B_particle[:, :] = (B_tmp[:, x_index_half].T * cx2 \
                     + B_tmp[:, (x_index_half+1)%n_x].T * cx1 \
                    ).T
  
    v = buneman_boris_v(c, dt, q, m, E_particle, B_particle, v)

    return v


def time_evolution_x(c, dt, v, x):
    
    x = buneman_boris_x(c, dt, v, x)

    return x 


def time_evolution_E(B, current, c, epsilon0, dx, dt, E):
    E[0, :] += -current[0, :] / epsilon0 * dt
    E[1, :] += (-current[1, :] / epsilon0 \
            - c**2 * (B[2, :] - np.roll(B[2, :], 1)) / dx) * dt 
    E[2, :] += (-current[2, :] / epsilon0 \
            + c**2 * (B[1, :] - np.roll(B[1, :], 1)) / dx) * dt
    return E


def time_evolution_B(E, dx, dt, B):
    #B[0, :] = B[0, :]
    B[1, :] += -(-(np.roll(E[2, :], -1) - E[2, :]) / dx) * dt 
    B[2, :] += -((np.roll(E[1, :], -1) - E[1, :]) / dx) * dt
    return B


def refrective_condition_x_left(v, x, x_min):

    over_xmax_index = np.where(x[0, :] < x_min)[0]
    x[0, over_xmax_index] = -x[0, over_xmax_index] + x_min
    v[0, over_xmax_index] = -v[0, over_xmax_index]

    return v, x


def refrective_condition_x_right(v, x, x_max):

    over_xmax_index = np.where(x[0, :] > x_max)[0]
    x[0, over_xmax_index] = 2.0 * x_max - x[0, over_xmax_index]
    v[0, over_xmax_index] = -v[0, over_xmax_index]

    return v, x


def open_condition_x_left(v_pic, x_pic, x_min):

    delete_index = np.where(x_pic[0, :] <= x_min) 
    x_pic = np.delete(x_pic, delete_index, axis=1)
    v_pic = np.delete(v_pic, delete_index, axis=1)

    return v_pic, x_pic


def open_condition_x_right(v_pic, x_pic, x_max):

    delete_index = np.where(x_pic[0, :] >= x_max) 
    x_pic = np.delete(x_pic, delete_index, axis=1)
    v_pic = np.delete(v_pic, delete_index, axis=1)

    return v_pic, x_pic


def boundary_B(B_pic):
    B_pic[:, 0] = B_pic[:, 1]
    B_pic[:, -1] = B_pic[:, -2]

    return B_pic


def boundary_E(E_pic):
    E_pic[0, 0] = 0.0
    E_pic[[1, 2], 0] = E_pic[[1, 2], 1]
    E_pic[0, [-1, -2]] = 0.0
    E_pic[[1, 2], -1] = E_pic[[1, 2], -2]

    return E_pic


def filter_E(rho, dx_pic, dt_pic, d_pic, epsilon0, E_pic):
    F = np.zeros(rho.shape)
    F = (E_pic[0, :] - np.roll(E_pic[0, :], 1, axis=0)) / dx_pic \
      - rho / epsilon0

    E_pic[0, :] += d_pic * (np.roll(F, -1, axis=0) - F) / dx_pic * dt_pic
    E_pic[0, 0] = E_pic[0, 1]
    E_pic[0, -1] = E_pic[0, -2]

    return E_pic