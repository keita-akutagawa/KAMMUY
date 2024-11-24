import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import cv2


# PIC
c = 1.0
mu_0 = 1.0
epsilon0 = 1.0 / (mu_0 * c**2)
m_electron = 1.0
r_m = 1 / 25
m_ion = m_electron / r_m
t_r = 1.0
ne0_pic = 50 #ここは手動で調整すること
ni0_pic = ne0_pic
B0_pic = np.sqrt(ne0_pic) / 1.0
Ti_pic  = (B0_pic**2 / 2.0 / mu_0) / (ni0_pic + ne0_pic * t_r)
Te_pic = Ti_pic * t_r
q_electron = -1 * np.sqrt(epsilon0 * Te_pic / ne0_pic)
q_ion = -1 * q_electron
debye_length = np.sqrt(epsilon0 * Te_pic / ne0_pic / q_electron**2)
omega_pe = np.sqrt(ne0_pic * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(ni0_pic * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0_pic / m_electron
omega_ci = q_ion * B0_pic / m_ion
VA_pic = B0_pic / np.sqrt(mu_0 * ni0_pic * m_ion)
gamma_pic = 5.0 / 3.0
rho_pic = ni0_pic * m_ion + ne0_pic * m_electron
p_pic = ni0_pic * Ti_pic + ne0_pic * Te_pic
CS_pic = np.sqrt(gamma_pic * p_pic / rho_pic)
v_thermal_electron = np.sqrt(2.0 * Te_pic / m_electron)
v_thermal_ion = np.sqrt(2.0 * Ti_pic / m_ion)
ion_inertial_length = c / omega_pi
electron_inertial_length = c / omega_pe
sheat_thickness = 1.0 * ion_inertial_length

dx_pic = 1.0
nx_pic = 20000
x_min_pic = 1e-10
x_max_pic = nx_pic * dx_pic - 1e-10
x_coordinate_pic = np.arange(0.0, x_max_pic, dx_pic)
dy_pic = 1.0
ny_pic = 100
y_min_pic = 1e-10
y_max_pic = ny_pic * dy_pic - 1e-10
y_coordinate_pic = np.arange(0.0, y_max_pic, dy_pic)

# MHD
gamma_mhd = 5.0 / 3.0
B0_mhd = B0_pic
rho0_mhd = ne0_pic * m_electron + ni0_pic * m_ion
p0_mhd = ne0_pic * Te_pic + ni0_pic * Ti_pic
VA_mhd = B0_mhd / np.sqrt(rho0_mhd)
CS_mhd = np.sqrt(gamma_mhd * p0_mhd / rho0_mhd)
Cf_mhd = np.sqrt(VA_mhd**2 + CS_mhd**2)
beta_mhd = p0_mhd / (B0_mhd**2 / 2)

dx_mhd = dx_pic
nx_mhd = nx_pic
x_min_mhd = 0.0
x_max_mhd = nx_mhd * dx_mhd
dy_mhd = dy_pic
ny_mhd = 2000
y_min_mhd = 0.0
y_max_mhd = ny_mhd * dy_mhd
x_coordinate_mhd = np.arange(x_min_mhd, x_max_mhd, dx_mhd)
y_coordinate_mhd = np.arange(y_min_mhd, y_max_mhd, dy_mhd)

# Interface
window_size = 5
interface_width = 20


# load data
procs = 8
buffer = 3
dirname = ""
step = 1200
savename = f"{step}_pic.png"

total_field_pic = np.zeros([nx_pic, ny_pic])
for rank in range(procs):
    local_grid_x = rank 
    local_grid_y = 0
    local_nx_pic = nx_pic // procs
    local_ny_pic = ny_pic // 1
    local_nx_mhd = nx_mhd // procs
    local_ny_mhd = ny_mhd // 1

    filename = f"{dirname}/current_sheet_large_B_{step}_{rank}.bin"
    with open(filename, 'rb') as f:
        B = np.fromfile(f, dtype=np.float32)
    B = B.reshape(local_nx_pic, local_ny_pic, 3).T
    filename = f"{dirname}/current_sheet_large_E_{step}_{rank}.bin"
    with open(filename, 'rb') as f:
        E = np.fromfile(f, dtype=np.float32)
    E = E.reshape(local_nx_pic, local_ny_pic, 3).T
    filename = f"{dirname}/current_sheet_large_current_{step}_{rank}.bin"
    with open(filename, 'rb') as f:
        current = np.fromfile(f, dtype=np.float32)
    current = current.reshape(local_nx_pic, local_ny_pic, 3).T

    filename = f"{dirname}/current_sheet_large_zeroth_moment_ion_{step}_{rank}.bin"
    with open(filename, 'rb') as f:
        zeroth_moment_ion = np.fromfile(f, dtype=np.float32)
    zeroth_moment_ion = zeroth_moment_ion.reshape(local_nx_pic, local_ny_pic).T
    filename = f"{dirname}/current_sheet_large_zeroth_moment_electron_{step}_{rank}.bin"
    with open(filename, 'rb') as f:
        zeroth_moment_electron = np.fromfile(f, dtype=np.float32)
    zeroth_moment_electron = zeroth_moment_electron.reshape(local_nx_pic, local_ny_pic).T

    rho_pic = m_ion * zeroth_moment_ion + m_electron * zeroth_moment_electron

    total_field_pic[
        int(local_grid_x * local_nx_pic) : int((local_grid_x + 1) * local_nx_pic), 
        int(local_grid_y * local_ny_pic) : int((local_grid_y + 1) * local_ny_pic)
    ] = rho_pic.T


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

X_pic, Y_pic = np.meshgrid(x_coordinate_pic, y_coordinate_pic - 0.5 * y_max_pic) / ion_inertial_length
mappable = ax1.pcolormesh(X_pic, Y_pic, total_field_pic.T / rho0_mhd, cmap='jet', vmin=0.0, vmax=2.0)

cbar = fig.colorbar(mappable, ax=ax1, pad=0.05, shrink=0.5, aspect=10, ticks=np.arange(0.0, 2.1, 0.5))
cbar.set_label(r'$\rho / \rho_0$', fontsize=24)
cbar.ax.tick_params(labelsize=18)

ax1.set_xlim(980, 1020)
ax1.set_ylim(-ny_pic / 2 / ion_inertial_length, ny_pic / 2 / ion_inertial_length)
ax1.set_xlabel(r'$x / \lambda_i$', fontsize=20)
ax1.set_ylabel(r'$y / \lambda_i$', fontsize=20)
ax1.tick_params(labelsize=18)
ax1.set_aspect("equal")

plt.tight_layout()
fig.savefig(savename, dpi=200)


