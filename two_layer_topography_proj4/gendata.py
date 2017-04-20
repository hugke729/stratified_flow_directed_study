from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from MyMITgcmUtils import write_for_mitgcm, create_size_header
from MyGrids import Grid, telescope_centre_n


def create_grid(r, D):
    # Grid (depends on flow configuration)
    Nx = 200
    Nz = 100
    dx = telescope_centre_n(
        dist=100e3, dx_min=200, x_centre=55e3, dx_const=3e3, n=Nx, init_r=1.1)
    dz = telescope_centre_n(
        dist=D, dx_min=1, x_centre=D*r, dx_const=2, n=Nz)
    g = Grid(dx, 1e3, dz)
    return g


def plot_grid():
    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(g.xc_km, g.dx)
    axs[0].set(xlabel='Distance (km)', ylabel='Horizontal grid spacing (m)')
    axs[1].plot(g.dz, g.zc)
    axs[1].set(ylabel='Depth (m)', xlabel='Vertical grid spacing (m)')
    axs[1].invert_yaxis()


def get_lower_layer_inds():
    return g.zf.max() - g.zc < r*D


def define_Tref():
    G = 9.81
    gprime = 1E-3*G  # reduced gravity
    alpha = 2E-4
    delta_temp = gprime/(G*alpha)
    T_ref = np.ones_like(g.zc)*15
    T_ref[lower_layer] -= delta_temp
    T_ref = gaussian_filter1d(T_ref, 1)
    return gprime, T_ref, delta_temp


def create_T0(regime):
    if regime == 'uni':
        T0 = T_ref*np.ones((g.Nx, g.Nz))
    if regime == 'exc':
        T0 = np.max(T_ref)*np.ones((g.Nx, g.Nz))
        left_ind = np.where(np.diff(topo) < 0)[0][0]
        T0[:left_ind, g.zc > Hm*(1-r)*D] -= delta_temp
        T0 = gaussian_filter(T0, (3, 1))
    return T0


def create_topo(Hm, regime):
    """Smoothed Gaussian with width scale of 10--20 km"""
    if regime == 'uni':
        topo = Hm*r*D*gaussian(g.Nx, 10) - D
    elif regime == 'exc':
        topo = Hm*r*D*gaussian(g.Nx, 30) - D
    return topo


def plot_init_conds():
    fig, ax = plt.subplots()
    cax = ax.pcolormesh(g.xf_km, g.zf, T0.T)
    ax.set(xlabel='Distance (km)', ylabel='Depth (m)')
    cbar = fig.colorbar(cax)
    cbar.set_label('Initial temperature field')

    ax.fill_between(g.xc_km, -topo, -topo.min(), color='grey')
    ax.set(xlabel='Distance (km)', ylabel='Depth (m)',
           xlim=g.xf_km[[0, -1]], ylim=g.zf[[-1, 0]],
           title=run_name)

    x_mid_ind = np.where(g.xc > g.xf[-1]/2)[0][0]
    U_mid = U0[x_mid_ind, ::3]
    z_ones = np.ones_like(g.zc[::3])
    ax.quiver(g.xc_km[x_mid_ind]*z_ones, g.zc[::3], U_mid, 0*z_ones)


def create_Uin(regime):
    """Create input velocity field for either unidirectional or exchange flow
    regime = is either 'uni' or 'exc' """
    if regime == 'uni':
        U_in = Fr*np.sqrt(r*(1-r)*gprime*D)*np.ones(g.Nz)
    elif regime == 'exc':
        U_in = np.zeros(g.Nz)
    return U_in


def create_U0():
    U0 = U_in*np.ones((g.Nx, g.Nz))
    return U0


def define_dirs():
    main_dir = ('/home/hugke729/mitgcm/stratified_flow_directed_study/'
                'two_layer_topography_proj4/')
    return {key: main_dir + key + '/' for key in ['code', 'input', 'runs']}


def write_inputs(run_name):

    inputs = dict(delXvar=g.dx, delZvar=g.dz, topo=topo, T0=T0, Tref=T_ref,
                  U_in=U_in, U0=U0, TW=T0[0, :], TE=T0[-1, :])
    run_dir = model_dirs['runs'] + run_name + '/'
    makedirs(run_dir, exist_ok=True)

    for k, v in inputs.items():
        write_for_mitgcm(run_dir + k + '.bin', v)


def write_summary(run_name):
    run_dir = model_dirs['runs'] + run_name + '/'
    with open(run_dir + 'summary.txt', 'wt') as f:
        summary_str = 'r: {0}, Hm: {1:2.2f}, Fr: {2}'.format(r, Hm, Fr)
        print(summary_str, file=f)


def write_sizeH():
    n = dict(nPx=1, nSx=1, sNx=g.Nx, OLx=2,
             nPy=1, nSy=1, sNy=g.Ny, OLy=2,
             Nr=g.Nz)
    create_size_header(model_dirs['code'], n)


if __name__ == '__main__':
    runs = dict(
        # r, Hm, Fr
        # Unidirectional
        uni_r_0p2_Hm_1_Fr_0p2=[0.2, 1, 0.2],
        uni_r_0p2_Hm_1_Fr_0p8=[0.2, 1, 0.8],
        uni_r_0p2_Hm_0p2_Fr_1=[0.2, 0.2, 1],
        uni_r_0p35_Hm_1_Fr_0p1=[0.35, 1, 0.1],
        uni_r_0p35_Hm_1_Fr_0p5=[0.35, 1, 0.5],
        uni_r_0p35_Hm_1_Fr_1=[0.35, 1, 1],
        uni_r_0p5_Hm_1_Fr_0p15=[0.5, 1, 0.15],
        uni_r_0p5_Hm_0p8_Fr_0p4=[0.5, 0.8, 0.4],
        uni_r_0p5_Hm_1_Fr_0p8=[0.5, 1, 0.8],
        uni_r_0p8_Hm_0p25_Fr_0p2=[0.8, 0.25, 0.2],
        uni_r_0p8_Hm_0p25_Fr_0p8=[0.8, 0.25, 0.8],
        # Exchange (Fr is not applicable, topography is always 0.6*D)
        exc1=[0.7, 0.6/0.7, None],
        exc2=[0.8, 0.6/0.8, None],
        exc3=[0.95, 0.6/0.95, None])

    D = 200

    for run_name, (r, Hm, Fr) in runs.items():
        regime = run_name[:3]
        # if regime != 'exc':
        #     continue
        g = create_grid(r, D)
        # plot_grid()
        lower_layer = get_lower_layer_inds()
        gprime, T_ref, delta_temp = define_Tref()
        topo = create_topo(Hm, regime)
        T0 = create_T0(regime)
        U_in = create_Uin(regime)
        U0 = create_U0()
        # if regime == 'exc':
        plot_init_conds()
        model_dirs = define_dirs()
        write_inputs(run_name)
        write_summary(run_name)

    write_sizeH()
