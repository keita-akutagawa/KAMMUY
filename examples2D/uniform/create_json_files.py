import numpy as np
import json 


buffer_mhd = 4
buffer_pic = 4
smr_avail = False 
number_of_levels = 1 

EPS = 1e-20
PI = np.pi 

c = 1.0
epsilon0 = 1.0
mu0 = 1.0 
dcoef_lm = 0.001 

m_ion = 25.0
m_electron = 1.0 
number_density_ion = 20
number_density_electron = number_density_ion 
rho0 = m_ion * number_density_ion + m_electron * number_density_electron 
B0 = np.sqrt(number_density_electron) / 1.0 
beta = 0.25 
p0 = beta * B0**2 / 2 / mu0
Te = beta * (B0**2 / 2 / mu0) / (number_density_ion + number_density_electron)
Ti = Te 
p0 = number_density_ion * Ti + number_density_electron * Te
q_electron = -np.sqrt(epsilon0 * Te / number_density_electron)
q_ion = -q_electron 
omega_pe = np.sqrt(number_density_electron * q_electron**2 / m_electron / epsilon0)
vthe = np.sqrt(Te / m_electron)
debye_length = vthe / omega_pe
omega_pi = np.sqrt(number_density_ion * q_ion**2 / m_ion / epsilon0)
ion_inertial_length = c / omega_pi

dx_pic = 1.0
dy_pic = 1.0
nx_pic = int(10 * ion_inertial_length + 2 * buffer_pic)
ny_pic = int(10 * ion_inertial_length + 2 * buffer_pic)

exist_num_ion = number_density_ion * nx_pic * ny_pic
exist_num_electron = exist_num_ion 
total_num_ion = exist_num_ion * 2 
total_num_electron = total_num_ion

dt_pic = min(0.1 / omega_pe, 0.5 / c)

grid_size_ratio = int(0.5 * ion_inertial_length)

dx_mhd = [grid_size_ratio * dx_pic]
dy_mhd = [grid_size_ratio * dy_pic]
nx_mhd = [int(20 * ion_inertial_length / dx_mhd[0]) + 2 * buffer_mhd]
ny_mhd = [int(20 * ion_inertial_length / dy_mhd[0]) + 2 * buffer_mhd]
start_index_x = [0]
start_index_y = [0]
end_index_x = [nx_mhd[0]]
end_index_y = [ny_mhd[0]] 
xmin_mhd = [-(nx_mhd[0]) / 2 * dx_mhd[0]] 
ymin_mhd = [-(ny_mhd[0]) / 2 * dy_mhd[0]]  
xmax_mhd = [nx_mhd[0] * dx_mhd[0] + xmin_mhd[0]]
ymax_mhd = [ny_mhd[0] * dy_mhd[0] + ymin_mhd[0]]

xmin_pic = -nx_pic / 2 * dx_pic
ymin_pic = -ny_pic / 2 * dy_pic
xmax_pic = nx_pic * dx_pic + xmin_pic 
ymax_pic = ny_pic * dy_pic + ymin_pic

start_index_in_mhd_x = int(nx_mhd[0] / 2 - nx_pic / 2 / grid_size_ratio) 
start_index_in_mhd_y = int(ny_mhd[0] / 2 - ny_pic / 2 / grid_size_ratio) 

convolution_interval = 5
delta_interlocking_function = 5 * dx_pic

nug_coef = 1e20
activate_isotropic_effect = True 
activate_gyrotropic_effect = False 
activate_hall_effect = False      

cr = 1.0
c_diff = 0.1

record_step = 10
total_step = 1000

addname = f"_isotropic"
save_dirname = f"/mnt/hdd0/akutagawa/KAMMUY_10momentMHD-PIC/results_uniform{addname}"
save_filename_without_step = f"uniform{addname}"


#parameter check start 

print(f"debye length = {debye_length}, dt_pic = {dt_pic}, ion_inertial_length = {ion_inertial_length}")
print(f"PIC grid: {nx_pic} X {ny_pic}, dx_pic = {dx_pic}, dy_pic = {dy_pic}")
print(f"xmin_pic = {xmin_pic}, xmax_pic = {xmax_pic}, ymin_pic = {ymin_pic}, ymax_pic = {ymax_pic}")
print(f"MHD grid = {nx_mhd[0]} X {ny_mhd[0]}, dx_mhd = {dx_mhd[0]}, dy_mhd = {dy_mhd[0]}")
print(f"xmin_mhd = {xmin_mhd[0]}, xmax_mhd = {xmax_mhd[0]}, ymin_mhd = {ymin_mhd[0]}, ymax_mhd = {ymax_mhd[0]}")
print(f"start_index_in_mhd_x = {start_index_in_mhd_x}, start_index_in_mhd_y = {start_index_in_mhd_y}")
print(f"grid size ratio = {grid_size_ratio}")
print("--------------------")

#parameter check end 


const_data_mhd = {
    "EPS": EPS, 
    "PI": PI,

    "nug_coef": nug_coef, 

    "activate_isotropic_effect": activate_isotropic_effect, 
    "activate_gyrotropic_effect": activate_gyrotropic_effect, 
    "activate_hall_effect": activate_hall_effect,  

    "cr": cr,

    "c_diff": c_diff, 

    "rho0": rho0, 
    "B0": B0, 
    "p0": p0, 

    "m_ion": m_ion, 
    "m_electron": m_electron, 
    "q_electron": q_electron, 

    "record_step": record_step, 
    "total_step": total_step, 

    "save_dirname": save_dirname, 
    "save_filename_without_step": save_filename_without_step
}

config_data_mhd = {
    "pusher": "ssprk3", 
    "reconstructor": "muscl", 
    "order": 2, 
    "boundary": {
        "x_left": "periodic",
        "x_right": "periodic", 
        "y_down": "periodic", 
        "y_up": "periodic", 
    },
    "smr_boundary": {
        "x_left": "interpolate", 
        "x_right": "interpolate", 
        "y_down": "interpolate", 
        "y_up": "interpolate", 
    },
    "output": "binary"
}

grid_data_mhd = {
    "buffer": buffer_mhd, 

    "nx": nx_mhd, 
    "ny": ny_mhd, 

    "dx": dx_mhd, 
    "dy": dy_mhd, 

    "xmin": xmin_mhd, 
    "ymin": ymin_mhd, 
    "xmax": xmax_mhd, 
    "ymax": ymax_mhd, 

    "start_index_x": start_index_x, 
    "start_index_y": start_index_y, 

    "end_index_x": end_index_x, 
    "end_index_y": end_index_y, 

    "smr_avail": smr_avail,  
    "number_of_levels": number_of_levels
}

const_data_pic = {
    "c": c, 
    "epsilon0": epsilon0, 
    "mu0": mu0, 
    "dcoef_lm": dcoef_lm, 
    "EPS": EPS, 
    "PI": PI,

    "exist_num_ion": exist_num_ion, 
    "exist_num_electron": exist_num_electron, 

    "total_num_ion": total_num_ion, 
    "total_num_electron": total_num_electron, 

    "m_ion": m_ion, 
    "m_electron": m_electron, 
    "q_ion": q_ion, 
    "q_electron": q_electron, 
    "number_density_ion": number_density_ion, 
    "number_density_electron": number_density_electron, 
    "B0": B0, 
    "p0": p0, 

    "omega_pe": omega_pe, 

    "record_step": record_step, 
    "total_step": total_step, 

    "dt": dt_pic, 

    "save_dirname": save_dirname, 
    "save_filename_without_step": save_filename_without_step
}

grid_data_pic = {
    "buffer": buffer_pic, 

    "nx": nx_pic, 
    "ny": ny_pic, 

    "dx": dx_pic, 
    "dy": dy_pic, 

    "xmin": xmin_pic, 
    "ymin": ymin_pic, 
    "xmax": xmax_pic, 
    "ymax": ymax_pic, 
}

const_data_interface = {
    "EPS": EPS, 
    "PI": PI,

    "convolution_interval": convolution_interval, 

    "delta_interlocking_function": delta_interlocking_function, 
}

grid_data_interface = {
    "grid_size_ratio": grid_size_ratio, 

    "start_index_in_mhd_x": start_index_in_mhd_x, 
    "start_index_in_mhd_y": start_index_in_mhd_y, 
}


with open("const_mhd.json", "w", encoding="utf-8") as f:
    json.dump(const_data_mhd, f, indent=4)

with open("config_mhd.json", "w", encoding="utf-8") as f:
    json.dump(config_data_mhd, f, indent=4)

with open("grid_mhd.json", "w", encoding="utf-8") as f:
    json.dump(grid_data_mhd, f, indent=4)

with open("const_pic.json", "w", encoding="utf-8") as f:
    json.dump(const_data_pic, f, indent=4)

with open("grid_pic.json", "w", encoding="utf-8") as f:
    json.dump(grid_data_pic, f, indent=4)

with open("const_interface.json", "w", encoding="utf-8") as f:
    json.dump(const_data_interface, f, indent=4)
\
with open("grid_interface.json", "w", encoding="utf-8") as f:
    json.dump(grid_data_interface, f, indent=4)