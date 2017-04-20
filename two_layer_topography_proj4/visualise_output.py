# Script to animate the evolution of the upstream hydraulic jump

from matplotlib.animation import FuncAnimation
from xmitgcm import open_mdsdataset
from MyFunctions import get_contour

run_dir = ('/home/hugke729/mitgcm/stratified_flow_directed_study/'
           'two_layer_topography_proj4/runs/uni_r_0p8_Hm_0p25_Fr_0p8/')

T_or_U = 'U'

ds = open_mdsdataset(run_dir, prefix=['T', 'U', 'Eta']).squeeze()

fig, ax = plt.subplots()

if T_or_U == 'T':
    cax = ds['T'].isel(time=0).plot.pcolormesh(ax=ax, vmin=10, vmax=15, cmap='jet')
elif T_or_U == 'U':
    if 'exc' in run_dir:
        cax = ds['U'].isel(time=0).plot.pcolormesh(
            ax=ax, vmin=-0.3, vmax=0.3, cmap='RdBu')
    elif 'uni' in run_dir:
        cax = ds['U'].isel(time=0).plot.pcolormesh(
            ax=ax, vmin=0, vmax=0.8, cmap='afmhot_r')
interface = get_contour(ds.XC, ds.Z, ds['T'].isel(time=0).values, 13.5)
ax.fill_between(ds.XC, -ds.Depth, -ds.Depth.values.max(), color='grey')
line, = ax.plot(ds.XC, interface)

ax.set(ylim=(-200, -0))
ax.axvline(ds.XG[10], color='grey')
ax.axvline(ds.XG[-10], color='grey')


def animate(i):
    T_i = ds['T'].isel(time=i).values
    if T_or_U == 'T':
        cax.set_array(T_i.flatten())
    elif T_or_U == 'U':
        U_i = ds['U'].isel(time=i).values
        cax.set_array(U_i.flatten())
    line.set_ydata(get_contour(ds.XC, ds.Z, T_i, 13.5))
    ax.set_title(str(i))


anim = FuncAnimation(fig, animate, frames=ds['time'].size, interval=500)
