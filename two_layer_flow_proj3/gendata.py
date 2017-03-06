# Script to set up MITgcm model simulations for two-layer shear flow
from os import makedirs
from collections import OrderedDict
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from MyGrids import telescope_centre_n
from MyMITgcmUtils import write_for_mitgcm, create_size_header


show_init_conds = False
incl_topo = True  # If False, topography is flat, else bump


def starting_interface(x, d_10, A, w):
    """Smoothed tophat centred about 0 with amplitude A and width w"""
    interface = d_10 * np.ones_like(x)
    interface[np.logical_and(x > -w / 2, x < w / 2)] += A
    interface = gaussian_filter1d(interface, sigma=3)
    return interface


exps = OrderedDict([
    # No shear, effectively one-layer
    [1, dict(r=0.1, u_10=0, u_20=0, A=0.05)],
    # Same experiment, with configuration inverted
    [2, dict(r=0.9, u_10=0, u_20=0, A=-0.05)],
    # As for exp1, but with moderate barotropic flow
    [3, dict(r=0.1, u_10=0.3, u_20=0.3, A=0.05)],
    # As for exp3, but flows in opposite directions
    [4, dict(r=0.1, u_10=0.3, u_20=-0.3, A=0.05)],
    # As for exp4, but inverted interface
    [5, dict(r=0.1, u_10=0.3, u_20=-0.3, A=-0.05)],
    # Interface straddles 0.5D
    [6, dict(r=0.475, u_10=0, u_20=0, A=0.05)],
    # Like exp3, but with thick bottom layer
    [7, dict(r=0.475, u_10=0.3, u_20=0.3, A=0.05)],
    # Like exp4, but with thick bottom layer
    [8, dict(r=0.475, u_10=0.3, u_20=-0.3, A=0.05)]])


# Experiments that include a hydraulic jump
# These experiments use much of the same inputs as those above. Only difference
# is the addition of non-flat topography
if incl_topo:
    # For each value of ru, increase velocity monotonically, and then
    # determine A (interface perturbation and bump in seafloor) that is
    # necessary to induce upstream hydraulic jump (e.g., Fig 2.11 of Baines)
    exps = OrderedDict([
        ['ru_0p1_vel_0p1', dict(r=0.1, u_10=0.1, u_20=0.1, A=0.15)],
        ['ru_0p1_vel_0p2', dict(r=0.1, u_10=0.2, u_20=0.2, A=0.05)],
        ['ru_0p1_vel_0p3', dict(r=0.1, u_10=0.3, u_20=0.3, A=0.05)],
        ['ru_0p1_vel_0p35', dict(r=0.1, u_10=0.35, u_20=0.35, A=0.05)],
        ['ru_0p1_vel_0p4', dict(r=0.1, u_10=0.4, u_20=0.4, A=0.05)],
        ['ru_0p1_vel_0p5', dict(r=0.1, u_10=0.4, u_20=0.4, A=0.4)],
        ['ru_0p3_vel_0p2', dict(r=0.3, u_10=0.2, u_20=0.2, A=0.25)],
        ['ru_0p3_vel_0p3', dict(r=0.3, u_10=0.3, u_20=0.3, A=0.15)],
        ['ru_0p3_vel_0p4', dict(r=0.3, u_10=0.4, u_20=0.4, A=0.05)],
        ['ru_0p3_vel_0p5', dict(r=0.3, u_10=0.5, u_20=0.5, A=0.05)]])

# Directories
main_dir = ('/home/hugke729/mitgcm/stratified_flow_directed_study/'
            'two_layer_flow_proj3/')
input_dir = main_dir + 'input/'
code_dir = main_dir + 'code/'
runs_dir = main_dir + 'runs/'

# Constants
D = 200
g = 9.81
gprime = 1E-3 * g  # reduced gravity
alpha = 2E-4
delta_temp = gprime / (g * alpha)

for k_exp, v_exp in exps.items():

    run_dir = runs_dir + 'exp_' + str(k_exp) + '/'
    makedirs(run_dir, exist_ok=True)
    d_10 = D*v_exp['r']  # Depth of bottom layer far away
    d_20 = D - d_10  # Depth of top layer far away

    # Grid (depends on flow configuration)
    Nx = 200
    Nz = 100
    dx = telescope_centre_n(
        dist=100e3, dx_min=200, x_centre=50e3, dx_const=3e3, n=Nx, init_r=1.1)
    dx = 1e3 * np.ones(Nx)
    x = np.cumsum(dx) - np.sum(dx) / 2
    # dz = D/Nz*np.ones(Nz)
    dz = telescope_centre_n(
        dist=D, dx_min=0.7, x_centre=d_20, dx_const=2, n=Nz)
    z = -np.cumsum(dz)
    X, Z = np.meshgrid(x, z, indexing='ij')
    topo = D * np.ones(Nx)

    # Starting conditions
    A = D*v_exp['A']  # Amplitude of top hat function
    interface = starting_interface(x, d_10, A, 5e3) - D

    if incl_topo:
        topo -= starting_interface(x, d_10, A, 5e3) - d_10

    lower_layer = z < -d_20
    lower_layer2D = Z < interface[:, None] * np.ones((Nx, Nz))

    T_ref = delta_temp * np.ones(Nz)
    T_ref[lower_layer] = 0
    T_ref = gaussian_filter1d(T_ref, 1)

    T0 = delta_temp * np.ones((Nx, Nz))
    T0[lower_layer2D] = 0
    T0 = gaussian_filter(T0, (1.2, 0.6))

    u1 = v_exp['u_10'] * np.sqrt(gprime * D)
    u2 = v_exp['u_20'] * np.sqrt(gprime * D)

    U_in = u2 * np.ones(Nz)
    U_in[lower_layer] = u1
    U_in = gaussian_filter(U_in, 0.6)

    U0 = u2 * np.ones((Nx, Nz))
    U0[lower_layer2D] = u1
    U0 = gaussian_filter(U0, (1.2, 0.6))

    # Write to files
    inputs = dict(Uw=U_in, Ue=U_in, U0=U0, Tw=T_ref, Te=T_ref, topo=-topo,
                  delZvar=dz, delXvar=dx, TRef=T_ref, T0=T0)

    for k, v in inputs.items():
        write_for_mitgcm(run_dir + k + '.bin', v, prec=64)

    if show_init_conds:
        fig, axs = plt.subplots(ncols=2)
        for i, Q in enumerate([U0, T0]):
            cax = axs[i].pcolormesh(x, z, Q.T)
            fig.colorbar(cax, ax=axs[i])

n = dict(sNx=Nx // 2, sNy=1, OLx=3, OLy=3, nSx=1, nSy=1, nPx=2, nPy=1, Nr=Nz)
create_size_header(code_dir, n)

