import numpy as np
from scipy import stats 
from lib_MHDPIC1D.pic1d import open_condition_x_left


def interlocking_function(x_interface_coordinate):
    F = 0.5 * (1.0 + np.cos(np.pi * (x_interface_coordinate - x_interface_coordinate[0]) / (x_interface_coordinate[-1] - x_interface_coordinate[0])))
    return F


def get_interface_quantity_MHDtoPIC(F, q_mhd, q_pic):
    q_interface = F * q_mhd + (1.0 - F) * q_pic
    return q_interface


def get_interface_quantity_PICtoMHD(F, q_mhd, q_pic):
    q_interface = F * q_mhd + (1.0 - F) * q_pic
    return q_interface


def convolve_parameter(q, window_size):

    convolved_q = q.copy().astype(np.float64)
    if len(q.shape) == 1:  # ベクトルの場合
        tmp_q = np.convolve(q, np.ones(window_size) / window_size, mode="valid")
        convolved_q[window_size//2 : -window_size//2 + 1] = tmp_q
        convolved_q[:window_size//2] = convolved_q[window_size//2]
        convolved_q[-window_size//2:] = convolved_q[-window_size//2]
    elif len(q.shape) == 2:  # 行列の場合
        for i in range(q.shape[0]):
            tmp_q = np.convolve(q[i, :], np.ones(window_size) / window_size, mode="valid")
            convolved_q[i, window_size//2 : -window_size//2 + 1] = tmp_q
            convolved_q[i, :window_size//2] = convolved_q[i, window_size//2]
            convolved_q[i, -window_size//2:] = convolved_q[i, -window_size//2]

    return convolved_q


def send_MHD_to_PICinterface_B(
        index_interface_mhd_start, index_interface_mhd_end, 
        index_interface_pic_start, index_interface_pic_end, 
        F, F_half, 
        U, window_size, B_pic
    ):

    Bx_mhd = U[4, :].copy()
    By_mhd = U[5, :].copy()
    Bz_mhd = U[6, :].copy()

    #PICグリッドに合わせる
    Bx_mhd = Bx_mhd #CT法は1次元では使っていないのでこのまま
    By_mhd = 0.5 * (By_mhd + np.roll(By_mhd, -1, axis=0))
    Bz_mhd = 0.5 * (Bz_mhd + np.roll(Bz_mhd, -1, axis=0))

    B_pic_tmp = B_pic.copy()
    B_pic_tmp = convolve_parameter(B_pic_tmp, window_size)
    
    Bx_mhd = Bx_mhd[index_interface_mhd_start:index_interface_mhd_end]
    By_mhd = By_mhd[index_interface_mhd_start:index_interface_mhd_end - 1]
    Bz_mhd = Bz_mhd[index_interface_mhd_start:index_interface_mhd_end - 1]
    Bx_pic = B_pic_tmp[0, index_interface_pic_start:index_interface_pic_end]
    By_pic = B_pic_tmp[1, index_interface_pic_start:index_interface_pic_end - 1]
    Bz_pic = B_pic_tmp[2, index_interface_pic_start:index_interface_pic_end - 1]
    
    B_pic[0, index_interface_pic_start:index_interface_pic_end] = get_interface_quantity_MHDtoPIC(
        F, Bx_mhd, Bx_pic
    )
    B_pic[1, index_interface_pic_start:index_interface_pic_end - 1] = get_interface_quantity_MHDtoPIC(
        F_half, By_mhd, By_pic
    )
    B_pic[2, index_interface_pic_start:index_interface_pic_end - 1] = get_interface_quantity_MHDtoPIC(
        F_half, Bz_mhd, Bz_pic
    )

    return B_pic


def send_MHD_to_PICinterface_E(
        index_interface_mhd_start, index_interface_mhd_end, 
        index_interface_pic_start, index_interface_pic_end, 
        F, F_half, 
        U, window_size, E_pic
    ):
    
    rho_mhd = U[0, :].copy()
    u_mhd = U[1, :].copy() / rho_mhd
    v_mhd = U[2, :].copy() / rho_mhd
    w_mhd = U[3, :].copy() / rho_mhd
    Bx_mhd = U[4, :].copy()
    By_mhd = U[5, :].copy()
    Bz_mhd = U[6, :].copy()
    Ex_mhd = -(v_mhd * Bz_mhd - w_mhd * By_mhd)
    Ey_mhd = -(w_mhd * Bx_mhd - u_mhd * Bz_mhd)
    Ez_mhd = -(u_mhd * By_mhd - v_mhd * Bx_mhd)

    #PICグリッドに合わせる
    Ex_mhd = 0.5 * (Ex_mhd + np.roll(Ex_mhd, -1, axis=0))

    E_pic_tmp = E_pic.copy()
    E_pic_tmp = convolve_parameter(E_pic_tmp, window_size)
    
    Ex_mhd = Ex_mhd[index_interface_mhd_start:index_interface_mhd_end - 1]
    Ey_mhd = Ey_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Ez_mhd = Ez_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Ex_pic = E_pic_tmp[0, index_interface_pic_start:index_interface_pic_end - 1]
    Ey_pic = E_pic_tmp[1, index_interface_pic_start:index_interface_pic_end]
    Ez_pic = E_pic_tmp[2, index_interface_pic_start:index_interface_pic_end]
    
    E_pic[0, index_interface_pic_start:index_interface_pic_end - 1] = get_interface_quantity_MHDtoPIC(
        F_half, Ex_mhd, Ex_pic
    )
    E_pic[1, index_interface_pic_start:index_interface_pic_end] = get_interface_quantity_MHDtoPIC(
        F, Ey_mhd, Ey_pic
    )
    E_pic[2, index_interface_pic_start:index_interface_pic_end] = get_interface_quantity_MHDtoPIC(
        F, Ez_mhd, Ez_pic
    )

    return E_pic


def send_MHD_to_PICinterface_current(
        index_interface_mhd_start, index_interface_mhd_end, 
        index_interface_pic_start, index_interface_pic_end, 
        F, F_half, 
        U, dx, window_size, current_pic
    ):
    
    Bx_mhd = U[4, :].copy()
    By_mhd = U[5, :].copy()
    Bz_mhd = U[6, :].copy()
    
    #PICグリッドに合わせる
    #jxは0だけど、一応書いておく
    current_x_mhd = np.zeros(Bx_mhd.shape)
    current_y_mhd = -(np.roll(Bz_mhd, -1, axis=0) - np.roll(Bz_mhd, 1, axis=0)) / (2*dx)
    current_z_mhd = (np.roll(By_mhd, -1, axis=0) - np.roll(By_mhd, 1, axis=0)) / (2*dx)
    current_y_mhd[0] = current_y_mhd[1] 
    current_y_mhd[-1] = current_y_mhd[-2] 
    current_z_mhd[0] = current_z_mhd[1] 
    current_z_mhd[-1] = current_z_mhd[-2] 
    current_x_mhd = 0.5 * (current_x_mhd + np.roll(current_x_mhd, -1, axis=0))
    current_x_mhd[0] = current_x_mhd[1] 
    current_x_mhd[-1] = current_x_mhd[-2] 

    current_pic_tmp = current_pic.copy()
    current_pic_tmp = convolve_parameter(current_pic_tmp, window_size)
    
    current_x_mhd = current_x_mhd[index_interface_mhd_start:index_interface_mhd_end - 1]
    current_y_mhd = current_y_mhd[index_interface_mhd_start:index_interface_mhd_end]
    current_z_mhd = current_z_mhd[index_interface_mhd_start:index_interface_mhd_end]
    current_x_pic = current_pic_tmp[0, index_interface_pic_start:index_interface_pic_end - 1]
    current_y_pic = current_pic_tmp[1, index_interface_pic_start:index_interface_pic_end]
    current_z_pic = current_pic_tmp[2, index_interface_pic_start:index_interface_pic_end]
    
    current_pic[0, index_interface_pic_start:index_interface_pic_end - 1] = get_interface_quantity_MHDtoPIC(
        F_half, current_x_mhd, current_x_pic
    )
    current_pic[1, index_interface_pic_start:index_interface_pic_end] = get_interface_quantity_MHDtoPIC(
        F, current_y_mhd, current_y_pic
    )
    current_pic[2, index_interface_pic_start:index_interface_pic_end] = get_interface_quantity_MHDtoPIC(
        F, current_z_mhd, current_z_pic
    )

    return current_pic



"""
def reset_particles(
        n_pic, bulk_speed_pic, v_th_squared_pic,
        index_interface_pic_start, index_interface_pic_end, F, 
        x_min_pic, dx, v_pic, x_pic
    ):

    for i in range(len(n_pic)):
        delete_index = np.where((x_pic[0, :] <= x_min_pic + (i + index_interface_pic_start + 1) * dx - 0.5 * dx) 
                                & (x_pic[0, :] > x_min_pic + (i + index_interface_pic_start) * dx - 0.5 * dx))[0]

        delete_num_particle = round(len(delete_index) * F[i])
        
        random_number = np.random.randint(1, 100000000)
        rs = np.random.RandomState(random_number)
        delete_index = rs.choice(delete_index, size=delete_num_particle, replace=False)
        
        x_pic = np.delete(x_pic, delete_index, axis=1)
        v_pic = np.delete(v_pic, delete_index, axis=1)
    
        reload_num_particle = round(n_pic[i] * F[i])

        new_particles_v = np.zeros([3, reload_num_particle])
        new_particles_x = np.zeros([3, reload_num_particle])
        random_number = np.random.randint(1, 100000000)
        new_particles_v[0, :] = np.asarray(
            stats.norm.rvs(bulk_speed_pic[0, i], np.sqrt(v_th_squared_pic[i]), size=reload_num_particle, random_state=random_number)
        )
        random_number = np.random.randint(1, 100000000)
        new_particles_v[1, :] = np.asarray(
            stats.norm.rvs(bulk_speed_pic[1, i], np.sqrt(v_th_squared_pic[i]), size=reload_num_particle, random_state=random_number)
        )
        random_number = np.random.randint(1, 100000000)
        new_particles_v[2, :] = np.asarray(
            stats.norm.rvs(bulk_speed_pic[2, i], np.sqrt(v_th_squared_pic[i]), size=reload_num_particle, random_state=random_number)
        )
        random_number = np.random.randint(1, 100000000)
        rs = np.random.RandomState(random_number)
        new_particles_x[0, :] = (rs.rand(reload_num_particle) - 0.5) * dx \
                              + x_min_pic + (index_interface_pic_start + i) * dx
        #new_particles_x[0, :] = (np.linspace(-0.49, 0.49, reset_num_particle)) * dx \
        #                      + (index_interface_pic_start + i) * dx
            
        v_pic = np.hstack([v_pic, new_particles_v])
        x_pic = np.hstack([x_pic, new_particles_x])
    
    v_pic, x_pic = open_condition_x_left(v_pic, x_pic, x_min_pic)

    return v_pic, x_pic
"""

def reset_particles(
        n_pic, bulk_speed_pic, v_th_squared_pic,
        index_interface_pic_start, index_interface_pic_end, F, 
        x_min_pic, dx, v_pic, x_pic
    ):

    particle_count = x_pic.shape[1]

    # 各インターフェースの削除対象インデックスをまとめて取得
    delete_indices = []
    for i in range(len(n_pic)):
        start_x = x_min_pic + (i + index_interface_pic_start) * dx - 0.5 * dx
        end_x = x_min_pic + (i + index_interface_pic_start + 1) * dx - 0.5 * dx
        indices = np.where((x_pic[0, :] > start_x) & (x_pic[0, :] <= end_x))[0]
        delete_num = round(len(indices) * F[i])
        if delete_num > 0:
            rs = np.random.RandomState(np.random.randint(1, 100000000))
            delete_indices.append(rs.choice(indices, size=delete_num, replace=False))

    # インデックスをフラットにして一括削除
    if delete_indices:
        delete_indices = np.concatenate(delete_indices)
        x_pic = np.delete(x_pic, delete_indices, axis=1)
        v_pic = np.delete(v_pic, delete_indices, axis=1)
    
    # パーティクルを一括で生成
    new_v_list = []
    new_x_list = []
    for i in range(len(n_pic)):
        reload_num = round(n_pic[i] * F[i])
        if reload_num > 0:
            random_state = np.random.RandomState(np.random.randint(1, 100000000))
            new_v = np.vstack([
                stats.norm.rvs(bulk_speed_pic[j, i], np.sqrt(v_th_squared_pic[i]), size=reload_num, random_state=random_state)
                for j in range(3)
            ])
            new_x = np.zeros((3, reload_num))
            new_x[0, :] = (random_state.rand(reload_num) - 0.5) * dx + x_min_pic + (index_interface_pic_start + i) * dx
            new_v_list.append(new_v)
            new_x_list.append(new_x)
    
    # 新しいパーティクルを追加
    if new_v_list:
        new_v_particles = np.hstack(new_v_list)
        new_x_particles = np.hstack(new_x_list)
        v_pic = np.hstack([v_pic, new_v_particles])
        x_pic = np.hstack([x_pic, new_x_particles])

    # 境界条件を適用
    v_pic, x_pic = open_condition_x_left(v_pic, x_pic, x_min_pic)

    return v_pic, x_pic

    

def send_MHD_to_PICinterface_particle(
        index_interface_mhd_start, index_interface_mhd_end, 
        index_interface_pic_start, index_interface_pic_end, 
        F, x_min_pic, 
        U, dx, gamma, q_electron, 
        m_electron, m_ion, nx_pic, c, window_size, 
        v_pic_ion, v_pic_electron, x_pic_ion, x_pic_electron
    ):

    rho_mhd = U[0, :]
    u_mhd = U[1, :] / rho_mhd
    v_mhd = U[2, :] / rho_mhd
    w_mhd = U[3, :] / rho_mhd
    Bx_mhd = U[4, :]
    By_mhd = U[5, :]
    Bz_mhd = U[6, :]
    e_mhd = U[7, :]
    p_mhd = (gamma - 1.0) \
          * (e_mhd - 0.5 * rho_mhd * (u_mhd**2+v_mhd**2+w_mhd**2)
              - 0.5 * (Bx_mhd**2+By_mhd**2+Bz_mhd**2))
    current_x_mhd = np.zeros(Bx_mhd.shape)
    current_y_mhd = -(np.roll(Bz_mhd, -1, axis=0) - np.roll(Bz_mhd, 1, axis=0)) / (2*dx)
    current_z_mhd = (np.roll(By_mhd, -1, axis=0) - np.roll(By_mhd, 1, axis=0)) / (2*dx)
    current_y_mhd[0] = current_y_mhd[1] 
    current_y_mhd[-1] = current_y_mhd[-2] 
    current_z_mhd[0] = current_z_mhd[1] 
    current_z_mhd[-1] = current_z_mhd[-2] 
    # ni = ne, Ti = Te のつもり
    ni_mhd = rho_mhd / (m_ion + m_electron)
    ne_mhd = ni_mhd
    Ti_mhd = p_mhd / 2.0 / ni_mhd
    Te_mhd = p_mhd / 2.0 / ne_mhd
    
    zeroth_moment_ion = np.zeros(nx_pic)
    zeroth_moment_electron = np.zeros(nx_pic)
    zeroth_moment_ion = get_zeroth_moment(x_pic_ion, nx_pic, dx, zeroth_moment_ion)
    zeroth_moment_electron = get_zeroth_moment(x_pic_electron, nx_pic, dx, zeroth_moment_electron)
    first_moment_ion = np.zeros([3, nx_pic])
    first_moment_electron = np.zeros([3, nx_pic])
    first_moment_ion = get_first_moment(c, v_pic_ion, x_pic_ion, nx_pic, dx, first_moment_ion)
    first_moment_electron = get_first_moment(c, v_pic_electron, x_pic_electron, nx_pic, dx, first_moment_electron)
    second_moment_ion = np.zeros([9, nx_pic])
    second_moment_electron = np.zeros([9, nx_pic])
    second_moment_ion = get_second_moment(c, v_pic_ion, x_pic_ion, nx_pic, dx, second_moment_ion)
    second_moment_electron = get_second_moment(c, v_pic_electron, x_pic_electron, nx_pic, dx, second_moment_electron)

    rho_pic = np.zeros(nx_pic)
    bulk_speed_pic = np.zeros([3, nx_pic])
    current_pic = np.zeros([3, nx_pic])

    rho_pic = m_ion * zeroth_moment_ion + m_electron * zeroth_moment_electron
    bulk_speed_pic[0, :] = (m_ion * first_moment_ion[0, :] + m_electron * first_moment_electron[0, :]) / rho_pic
    bulk_speed_pic[1, :] = (m_ion * first_moment_ion[1, :] + m_electron * first_moment_electron[1, :]) / rho_pic
    bulk_speed_pic[2, :] = (m_ion * first_moment_ion[2, :] + m_electron * first_moment_electron[2, :]) / rho_pic
    q_ion = -1.0 * q_electron
    current_pic[0, :] = q_ion * first_moment_ion[0, :] + q_electron * first_moment_electron[0, :]
    current_pic[1, :] = q_ion * first_moment_ion[1, :] + q_electron * first_moment_electron[1, :]
    current_pic[2, :] = q_ion * first_moment_ion[2, :] + q_electron * first_moment_electron[2, :]

    rho_pic = convolve_parameter(rho_pic, window_size)
    bulk_speed_pic = convolve_parameter(bulk_speed_pic, window_size)
    current_pic = convolve_parameter(current_pic, window_size)

    rho_pic = rho_pic[index_interface_pic_start:index_interface_pic_end]
    bulk_speed_pic = bulk_speed_pic[:, index_interface_pic_start:index_interface_pic_end]
    current_pic = current_pic[:, index_interface_pic_start:index_interface_pic_end]
    
    rho_mhd = rho_mhd[index_interface_mhd_start:index_interface_mhd_end]
    u_mhd = u_mhd[index_interface_mhd_start:index_interface_mhd_end]
    v_mhd = v_mhd[index_interface_mhd_start:index_interface_mhd_end]
    w_mhd = w_mhd[index_interface_mhd_start:index_interface_mhd_end]
    current_x_mhd = current_x_mhd[index_interface_mhd_start:index_interface_mhd_end]
    current_y_mhd = current_y_mhd[index_interface_mhd_start:index_interface_mhd_end]
    current_z_mhd = current_z_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Ti_mhd = Ti_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Te_mhd = Te_mhd[index_interface_mhd_start:index_interface_mhd_end]


    rho_pic = get_interface_quantity_MHDtoPIC(
        F, rho_mhd, rho_pic
    )
    bulk_speed_pic[0, :] = get_interface_quantity_MHDtoPIC(
        F, u_mhd, bulk_speed_pic[0, :]
    )
    bulk_speed_pic[1, :] = get_interface_quantity_MHDtoPIC(
        F, v_mhd, bulk_speed_pic[1, :]
    )
    bulk_speed_pic[2, :] = get_interface_quantity_MHDtoPIC(
        F, w_mhd, bulk_speed_pic[2, :]
    )
    current_pic[0, :] = get_interface_quantity_MHDtoPIC(
        F, current_x_mhd, current_pic[0, :]
    )
    current_pic[1, :] = get_interface_quantity_MHDtoPIC(
        F, current_y_mhd, current_pic[1, :]
    )
    current_pic[2, :] = get_interface_quantity_MHDtoPIC(
        F, current_z_mhd, current_pic[2, :]
    )

    ni_pic = rho_pic / (m_electron + m_ion)
    ne_pic = ni_pic
    v_thi_squared_pic = 2.0 * Ti_mhd / m_ion      
    v_the_squared_pic = 2.0 * Te_mhd / m_electron

    #v_thi_squared_pic = np.maximum(v_thi_squared_pic, np.zeros(v_thi_squared_pic.shape[0]) + 1e-10)
    #v_the_squared_pic = np.maximum(v_the_squared_pic, np.zeros(v_the_squared_pic.shape[0]) + 1e-10)

    bulk_speed_ion = np.zeros(bulk_speed_pic.shape)
    bulk_speed_ion = bulk_speed_pic
    v_pic_ion, x_pic_ion = reset_particles(
        ni_pic, bulk_speed_ion, v_thi_squared_pic,
        index_interface_pic_start, index_interface_pic_end, F, 
        x_min_pic, dx, v_pic_ion, x_pic_ion
    )
    bulk_speed_electron = np.zeros(bulk_speed_pic.shape)
    bulk_speed_electron[0, :] = bulk_speed_pic[0, :] - current_pic[0, :] / ne_pic / np.abs(q_electron)
    bulk_speed_electron[1, :] = bulk_speed_pic[1, :] - current_pic[1, :] / ne_pic / np.abs(q_electron)
    bulk_speed_electron[2, :] = bulk_speed_pic[2, :] - current_pic[2, :] / ne_pic / np.abs(q_electron)
    v_pic_electron, x_pic_electron = reset_particles(
        ne_pic, bulk_speed_electron, v_the_squared_pic,
        index_interface_pic_start, index_interface_pic_end, F, 
        x_min_pic, dx, v_pic_electron, x_pic_electron
    )
    
    return v_pic_ion, v_pic_electron, x_pic_ion, x_pic_electron


def send_PIC_to_MHDinterface(
        index_interface_mhd_start, index_interface_mhd_end, 
        index_interface_pic_start, index_interface_pic_end, 
        F, 
        gamma, m_electron, m_ion, nx_pic, B_pic, 
        zeroth_moment_ion, zeroth_moment_electron, 
        first_moment_ion, first_moment_electron, 
        second_moment_ion, second_moment_electron, 
        window_size, U
    ):

    #MHDグリッドに合わせる
    B_pic_tmp = B_pic.copy()
    Bx_pic_tmp = B_pic_tmp[0, :]
    By_pic_tmp = 0.5 * (B_pic_tmp[1, :] + np.roll(B_pic_tmp[1, :], 1, axis=0))
    Bz_pic_tmp = 0.5 * (B_pic_tmp[2, :] + np.roll(B_pic_tmp[2, :], 1, axis=0))

    rho_pic = np.zeros(nx_pic)
    bulk_speed_pic = np.zeros([3, nx_pic])
    v_thi_squared_pic = np.zeros(nx_pic)
    v_the_squared_pic = np.zeros(nx_pic)
    p_pic = np.zeros(nx_pic)

    rho_pic = m_ion * zeroth_moment_ion + m_electron * zeroth_moment_electron
    bulk_speed_pic[0, :] = (m_ion * first_moment_ion[0, :] + m_electron * first_moment_electron[0, :]) / rho_pic
    bulk_speed_pic[1, :] = (m_ion * first_moment_ion[1, :] + m_electron * first_moment_electron[1, :]) / rho_pic
    bulk_speed_pic[2, :] = (m_ion * first_moment_ion[2, :] + m_electron * first_moment_electron[2, :]) / rho_pic
    v_thi_squared_pic = ((second_moment_ion[0, :] + second_moment_ion[4, :] + second_moment_ion[8, :])
                        - (first_moment_ion[0, :]**2 + first_moment_ion[1, :]**2 + first_moment_ion[2, :]**2) / zeroth_moment_ion) \
                        / 3.0 / (zeroth_moment_ion + 1e-10)
    v_the_squared_pic = ((second_moment_electron[0, :] + second_moment_electron[4, :] + second_moment_electron[8, :])
                        - (first_moment_electron[0, :]**2 + first_moment_electron[1, :]**2 + first_moment_electron[2, :]**2) / zeroth_moment_electron) \
                        / 3.0 / (zeroth_moment_electron + 1e-10)
    p_pic = zeroth_moment_electron * m_electron * v_the_squared_pic / 2.0 + zeroth_moment_ion * m_ion * v_thi_squared_pic / 2.0
    
    rho_pic = convolve_parameter(rho_pic, window_size)
    bulk_speed_pic = convolve_parameter(bulk_speed_pic, window_size)
    Bx_pic_tmp = convolve_parameter(Bx_pic_tmp, window_size)
    By_pic_tmp = convolve_parameter(By_pic_tmp, window_size)
    Bz_pic_tmp = convolve_parameter(Bz_pic_tmp, window_size)
    p_pic = convolve_parameter(p_pic, window_size)

    rho_pic = rho_pic[index_interface_pic_start:index_interface_pic_end]
    bulk_speed_pic = bulk_speed_pic[:, index_interface_pic_start:index_interface_pic_end]
    Bx_pic = Bx_pic_tmp[index_interface_pic_start:index_interface_pic_end]
    By_pic = By_pic_tmp[index_interface_pic_start:index_interface_pic_end]
    Bz_pic = Bz_pic_tmp[index_interface_pic_start:index_interface_pic_end]
    p_pic = p_pic[index_interface_pic_start:index_interface_pic_end]

    rho_mhd = U[0, :]
    u_mhd = U[1, :] / rho_mhd
    v_mhd = U[2, :] / rho_mhd
    w_mhd = U[3, :] / rho_mhd
    Bx_mhd = U[4, :]
    By_mhd = U[5, :]
    Bz_mhd = U[6, :]
    e_mhd = U[7, :]
    p_mhd = (gamma - 1.0) \
          * (e_mhd - 0.5 * rho_mhd * (u_mhd**2+v_mhd**2+w_mhd**2)
              - 0.5 * (Bx_mhd**2+By_mhd**2+Bz_mhd**2))
    # ni = ne, Ti = Te のつもり
    ni_mhd = rho_mhd / (m_ion + m_electron)
    ne_mhd = ni_mhd
    Ti_mhd = p_mhd / 2.0 / ni_mhd
    Te_mhd = p_mhd / 2.0 / ne_mhd
    
    rho_mhd = rho_mhd[index_interface_mhd_start:index_interface_mhd_end]
    u_mhd = u_mhd[index_interface_mhd_start:index_interface_mhd_end]
    v_mhd = v_mhd[index_interface_mhd_start:index_interface_mhd_end]
    w_mhd = w_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Bx_mhd = Bx_mhd[index_interface_mhd_start:index_interface_mhd_end]
    By_mhd = By_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Bz_mhd = Bz_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Ti_mhd = Ti_mhd[index_interface_mhd_start:index_interface_mhd_end]
    Te_mhd = Te_mhd[index_interface_mhd_start:index_interface_mhd_end]


    rho_mhd = get_interface_quantity_PICtoMHD(F, rho_mhd, rho_pic)
    u_mhd = get_interface_quantity_PICtoMHD(F, u_mhd, bulk_speed_pic[0, :])
    v_mhd = get_interface_quantity_PICtoMHD(F, v_mhd, bulk_speed_pic[1, :])
    w_mhd = get_interface_quantity_PICtoMHD(F, w_mhd, bulk_speed_pic[2, :])
    Bx_mhd = get_interface_quantity_PICtoMHD(F, Bx_mhd, Bx_pic)
    By_mhd = get_interface_quantity_PICtoMHD(F, By_mhd, By_pic)
    Bz_mhd = get_interface_quantity_PICtoMHD(F, Bz_mhd, Bz_pic)

    ni_mhd = rho_mhd / (m_ion + m_electron)
    ne_mhd = ni_mhd
    p_mhd = ni_mhd * Ti_mhd + ne_mhd * Te_mhd

    U[0, index_interface_mhd_start:index_interface_mhd_end] = rho_mhd
    U[1, index_interface_mhd_start:index_interface_mhd_end] = u_mhd * rho_mhd
    U[2, index_interface_mhd_start:index_interface_mhd_end] = v_mhd * rho_mhd
    U[3, index_interface_mhd_start:index_interface_mhd_end] = w_mhd * rho_mhd
    U[4, index_interface_mhd_start:index_interface_mhd_end] = Bx_mhd
    U[5, index_interface_mhd_start:index_interface_mhd_end] = By_mhd
    U[6, index_interface_mhd_start:index_interface_mhd_end] = Bz_mhd
    e_mhd = p_mhd / (gamma - 1.0) + 0.5 * rho_mhd * (u_mhd**2 + v_mhd**2 + w_mhd**2) \
          + 0.5 * (Bx_mhd**2 + By_mhd**2 + Bz_mhd**2)
    U[7, index_interface_mhd_start:index_interface_mhd_end] = e_mhd

    return U



# PIC用

def get_zeroth_moment(x, n_x, dx, zeroth_moment):
    x_index = np.floor(x[0, :] / dx).astype(int)

    cx1 = (x[0, :] - x_index*dx)/dx  
    cx2 = ((x_index+1)*dx - x[0, :])/dx 
    index_one_array = x_index

    zeroth_moment[:] += np.bincount(index_one_array, 
                                 weights=cx2, 
                                 minlength=n_x
                                )
    zeroth_moment[:] += np.roll(np.bincount(index_one_array, 
                                         weights=cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    
    zeroth_moment[0] = zeroth_moment[1]
    zeroth_moment[-1] = zeroth_moment[-2]
    
    return zeroth_moment


def reload_get_zeroth_moment(x, n_x, dx, zeroth_moment):
    x_index = np.round(x[0, :] / dx - 1e-10).astype(int)
    x_index[x_index == n_x] = 0

    index_one_array = x_index

    zeroth_moment += np.bincount(index_one_array)

    zeroth_moment[0] = zeroth_moment[1]
    zeroth_moment[-1] = zeroth_moment[-2]
    
    return zeroth_moment


def get_first_moment(c, v, x, n_x, dx, first_moment):
    x_index = np.floor(x[0, :] / dx).astype(int)

    cx1 = (x[0, :] - x_index*dx)/dx  
    cx2 = ((x_index+1)*dx - x[0, :])/dx 
    index_one_array = x_index

    first_moment[0, :] += np.bincount(index_one_array, 
                                 weights=v[0, :] * cx2, 
                                 minlength=n_x
                                )
    first_moment[0, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[0, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    first_moment[1, :] += np.bincount(index_one_array, 
                                 weights=v[1, :] * cx2, 
                                 minlength=n_x
                                )
    first_moment[1, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[1, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    first_moment[2, :] += np.bincount(index_one_array, 
                                 weights=v[2, :] * cx2, 
                                 minlength=n_x
                                )
    first_moment[2, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[2, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    
    first_moment[:, 0] = first_moment[:, 1]
    first_moment[:, -1] = first_moment[:, -2]

    return first_moment


def reload_get_first_moment(c, v, x, n_x, dx, first_moment):
    x_index = np.round(x[0, :] / dx - 1e-10).astype(int)
    x_index[x_index == n_x] = 0

    index_one_array = x_index

    first_moment[0, :] += np.bincount(index_one_array, 
                                 weights=v[0, :], 
                                 minlength=n_x
                                )
    first_moment[1, :] += np.bincount(index_one_array, 
                                 weights=v[1, :], 
                                 minlength=n_x
                                )
    first_moment[2, :] += np.bincount(index_one_array, 
                                 weights=v[2, :], 
                                 minlength=n_x
                                )
    
    first_moment[:, 0] = first_moment[:, 1]
    first_moment[:, -1] = first_moment[:, -2]

    return first_moment


def get_second_moment(c, v, x, n_x, dx, second_moment):
    x_index = np.floor(x[0, :] / dx).astype(int)

    cx1 = (x[0, :] - x_index*dx)/dx  
    cx2 = ((x_index+1)*dx - x[0, :])/dx 
    index_one_array = x_index

    second_moment[0, :] += np.bincount(index_one_array, 
                                 weights=v[0, :] * v[0, :] * cx2, 
                                 minlength=n_x
                                )
    second_moment[0, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[0, :] * v[0, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    second_moment[1, :] += np.bincount(index_one_array, 
                                 weights=v[0, :] * v[1, :] * cx2, 
                                 minlength=n_x
                                )
    second_moment[1, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[0, :] * v[1, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    second_moment[2, :] += np.bincount(index_one_array, 
                                 weights=v[0, :] * v[2, :] * cx2, 
                                 minlength=n_x
                                )
    second_moment[2, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[0, :] * v[2, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    second_moment[3, :] = second_moment[1, :]
    second_moment[4, :] += np.bincount(index_one_array, 
                                 weights=v[1, :] * v[1, :] * cx2, 
                                 minlength=n_x
                                )
    second_moment[4, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[1, :] * v[1, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    second_moment[5, :] += np.bincount(index_one_array, 
                                 weights=v[1, :] * v[2, :] * cx2, 
                                 minlength=n_x
                                )
    second_moment[5, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[1, :] * v[2, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    second_moment[6, :] = second_moment[2, :]
    second_moment[7, :] = second_moment[5, :]
    second_moment[8, :] += np.bincount(index_one_array, 
                                 weights=v[2, :] * v[2, :] * cx2, 
                                 minlength=n_x
                                )
    second_moment[8, :] += np.roll(np.bincount(index_one_array, 
                                         weights=v[2, :] * v[2, :] * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    
    second_moment[:, 0] = second_moment[:, 1]
    second_moment[:, -1] = second_moment[:, -2]

    return second_moment


def reload_get_second_moment(c, v, x, n_x, dx, second_moment):
    x_index = np.round(x[0, :] / dx - 1e-10).astype(int)
    x_index[x_index == n_x] = 0
    index_one_array = x_index

    second_moment[0, :] += np.bincount(index_one_array, 
                                 weights=v[0, :] * v[0, :], 
                                 minlength=n_x
                                )
    second_moment[1, :] += np.bincount(index_one_array, 
                                 weights=v[0, :] * v[1, :], 
                                 minlength=n_x
                                )
    second_moment[2, :] += np.bincount(index_one_array, 
                                 weights=v[0, :] * v[2, :], 
                                 minlength=n_x
                                )
    second_moment[3, :] = second_moment[1, :]
    second_moment[4, :] += np.bincount(index_one_array, 
                                 weights=v[1, :] * v[1, :], 
                                 minlength=n_x
                                )
    second_moment[5, :] += np.bincount(index_one_array, 
                                 weights=v[1, :] * v[2, :], 
                                 minlength=n_x
                                )
    second_moment[6, :] = second_moment[2, :]
    second_moment[7, :] = second_moment[5, :]
    second_moment[8, :] += np.bincount(index_one_array, 
                                 weights=v[2, :] * v[2, :], 
                                 minlength=n_x
                                )
    
    second_moment[:, 0] = second_moment[:, 1]
    second_moment[:, -1] = second_moment[:, -2]

    return second_moment