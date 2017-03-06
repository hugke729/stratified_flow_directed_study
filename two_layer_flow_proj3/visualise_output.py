# Script to animate the evolution of the upstream hydraulic jump

from matplotlib.animation import FuncAnimation
from xmitgcm import open_mdsdataset
from MyFunctions import get_contour

run_dir = ('/home/hugke729/mitgcm/stratified_flow_directed_study/'
           'two_layer_flow_proj3/runs/exp_ru_0p3_vel_0p3/')

ds = open_mdsdataset(run_dir, prefix=['T', 'U'], delta_t=60).squeeze()

fig, ax = plt.subplots()
# cax = ds['T'].isel(time=0).plot.pcolormesh(ax=ax, vmin=0, vmax=5)
cax = ds['U'].isel(time=0).plot.pcolormesh(ax=ax, vmin=0, vmax=0.8, cmap='RdBu_r')
interface = get_contour(ds.XC, ds.Z, ds['T'].isel(time=0).values, 2.5)
line, = ax.plot(ds.XC, interface)

ax.set(ylim=(-200, -100))
ax.plot(2*(ds.XG[10], ), (-150, -200))
ax.plot(2*(ds.XG[-10], ), (-150, -200))


def animate(i):
    T_i = ds['T'].isel(time=i).values
    U_i = ds['U'].isel(time=i).values
    cax.set_array(U_i.flatten())
    line.set_ydata(get_contour(ds.XC, ds.Z, T_i, 2.5))
    ax.set_title(str(ds.time[i].values/3600))


anim = FuncAnimation(fig, animate, frames=ds['time'].size)
