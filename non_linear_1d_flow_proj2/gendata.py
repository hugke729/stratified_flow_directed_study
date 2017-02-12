# Recreate Baines (1995) figure 2.11 by using 2-layer flow
# This script creates the set up for the five regimes shown, and two extra
# regimes with Fr=0.8, 1.2

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from MyGrids import telescope_centre_n
from MyMITgcmUtils import write_for_mitgcm, create_size_header

for regime in ['sub', 'super', 'complete_block', 'partial_block_w_lee',
               'partial_block_no_lee', 'Fr0p8', 'Fr1p2']:

    model_dir = ('/home/hugke729/mitgcm/stratified_flow_directed_study/' +
                 'non_linear_1d_flow_proj2/')
    code_dir, runs_dir, input_dir = (
        model_dir + s for s in ['code/', 'runs/', 'input/'])

    # Create grid (telescope away from obstacle in x and z)

    # Let d0 be 100m deep and whole water column 1000.
    d0 = 100
    z_tot = 1000

    x_tot = 500e3
    dx_min = 3e3
    x_centre = 250e3

    # This telescoping function needs work. It's fragile
    # dist, dx_min, x_centre, dx_const, n, init_r
    dz = -telescope_centre_n(z_tot, 2, 100, 2, 80, 1.05)[::-1]
    dx = telescope_centre_n(x_tot, dx_min, x_centre, 8., 100, 1.05)

    # Create SIZE.h
    nx, nz = len(dx), len(dz)
    n = dict(sNx=nx//2, sNy=1, OLx=3, OLy=3, nSx=1, nSy=1, nPx=2, nPy=1, Nr=nz)
    create_size_header(code_dir, n)

    xf, zf = [np.insert(np.cumsum(arr), 0, 0) for arr in [dx, dz]]
    xc, zc = [(arr[:-1] + arr[1:])/2 for arr in [xf, zf]]

    g = 9.81
    gprime = 1E-3*g
    alpha = 2E-4
    delta_temp = gprime/(g*alpha)

    # Smoothed heaviside function
    T_ref = np.ones_like(dz)*delta_temp
    T_ref[zc < -(z_tot - d0)] = 0
    T_ref = gaussian_filter1d(T_ref, 0.5)

    # Let d_0 be 1
    # Obstacle is gaussian with different h_0 and Fr for different regime
    h0_Fr = dict(sub=(0.2, 0.2), super=(0.2, 2), complete_block=(2, 0.2),
                 partial_block_w_lee=(1, 0.2), partial_block_no_lee=(1, 1),
                 Fr0p8=(0.2, 0.8), Fr1p2=(0.2, 1.2))

    for regime, (h0, Fr) in h0_Fr.items():
        obstacle = h0*d0*np.exp(-(xc - x_centre)**2/(20e3)**2) - z_tot
        U_in = Fr*np.sqrt(gprime*d0)*np.ones(nz)

        T0 = T_ref[np.newaxis, :]*np.ones((nx, nz))
        U0 = U_in[np.newaxis, :]*np.ones((nx, nz))

        inputs = dict(Uw=U_in, Ue=U_in, U0=U0, Tw=T_ref, Te=T_ref, topo=obstacle,
                      delZvar=-dz, delXvar=dx, TRef=T_ref, T0=T0)

        for k, v in inputs.items():
            write_for_mitgcm(runs_dir + regime + '/' + k + '.bin', v, prec=64)
