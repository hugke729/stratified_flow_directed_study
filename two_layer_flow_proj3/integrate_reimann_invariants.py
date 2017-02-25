import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation


def interp(x, y):
    """Save me specifying extrapolate everytime"""
    return interp1d(x, y, fill_value='extrapolate')


def starting_interface(x, d_10, A, w):
    """Smoothed tophat centred about 0 with amplitude A and width w"""
    interface = d_10*np.ones_like(x)
    interface[np.logical_and(x > -w/2, x < w/2)] += A
    interface = gaussian_filter1d(interface, sigma=3)
    return interface


def calc_c(eta, v):
    """Calculate wavespeed (equivalently the characteristics)

    Divide into three parts, the three terms in Eq 3.3.14 of Baines (1998)"""
    c_1 = 0.0  # Actually Q/D but come back to this if needed
    c_2 = v*(d_20 - d_10 - 2*eta)/D
    c_3_sq = (g - v**2/D)*(d_20 - eta)*(d_10 + eta)/D
    # c_3_sq = c_3_sq.clip(min=0)
    c_plus = c_1 + c_2 + np.sqrt(c_3_sq)
    c_minus = c_1 + c_2 - np.sqrt(c_3_sq)

    return c_plus, c_minus


def calc_R(eta, v):
    """Calculate Riemann invariants

    Break into two terms, the terms in Equation 3.3.13 of Baines (1998)"""
    R_1_arg = (2*eta + d_10 - d_20)/D
    R_2_arg = v/np.sqrt(g*D)

    R_1 = np.arcsin(R_1_arg.clip(max=1))
    R_2 = np.arcsin(R_2_arg)

    R_plus = R_1 + R_2
    R_minus = R_1 - R_2

    return R_plus, R_minus


def solve_for_eta_v(R_plus, R_minus):
    """Solve system of Equations 3.3.13 of Baines (1998) for eta and v

    Two equations since plus and minus"""
    eta = 0.5*(D*np.sin((R_plus + R_minus)/2) - d_10 + d_20)
    v = np.sqrt(g*D)*np.sin((R_plus - R_minus)/2)

    return eta, v


if __name__ == '__main__':

    # Grid
    x = np.r_[-10:10:200j]
    delta_t = 0.1
    t = np.r_[0:50:delta_t]
    Nx, Nt = len(x), len(t)

    # Constants/initial values
    g = 0.1  # reduced gravity
    D = 1  # Total depth
    d_10 = 0.19  # Depth of bottom layer far away
    d_20 = D - d_10  # Depth of top layer far away
    A = 0.02

    # Preallocate output and fill zeroth time step
    eta_mat = np.empty((Nt + 1, Nx))
    v_mat = np.empty((Nt + 1, Nx))
    eta_mat[0] = starting_interface(x, d_10, A, w=3)
    v_mat[0] = np.zeros_like(x)

    for i, t in enumerate(t, start=1):

        # Values from previous time step
        eta_im1 = eta_mat[i-1]
        v_im1 = v_mat[i-1]
        R_plus_i, R_minus_i = calc_R(eta_im1, v_im1)
        R_plus = interp(x, R_plus_i)
        R_minus = interp(x, R_minus_i)

        # eta and v as functions of x
        eta_f = interp(x, eta_im1)
        v_f = interp(x, v_im1)

        # Project back along characteristics from t+1 to t
        c_plus, c_minus = calc_c(eta_im1, v_im1)
        x1 = x - c_plus*delta_t
        x2 = x - c_minus*delta_t

        # Calculate Riemann invariants
        R_plus_i = R_plus(x1)
        R_minus_i = R_minus(x2)

        eta_i, v_i = solve_for_eta_v(R_plus_i, R_minus_i)

        # Save current time step
        eta_mat[i] = eta_i
        v_mat[i] = v_i


fig, ax = plt.subplots()
ax.set(ylim=(0, 1))

step = 10
line, = ax.plot(x, eta_mat[0])


def animate(i):
    line.set_ydata(eta_mat[i*step])

anim = FuncAnimation(fig, animate, frames=Nt//step, interval=100)
