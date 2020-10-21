import numpy as np
from scipy.interpolate import interp1d

data = np.genfromtxt('Moser_Channel_ReTau590.csv',delimiter=',',comments='#',skip_header=5)
npts = data.shape[0]

ys = np.empty(npts*2-1)
ys[0:npts] = data[:,0]
ys[npts::] = 2.0 - np.flip(ys[0:npts-1])

Us = np.empty(npts*2-1)
Us[0:npts] = data[:,1]
Us[npts::] = np.flip(Us[0:npts-1])

Ruu = np.empty(npts*2-1)
Ruu[0:npts] = data[:,2]
Ruu[npts::] = np.flip(Ruu[0:npts-1])

Rvv = np.empty(npts*2-1)
Rvv[0:npts] = data[:,3]
Rvv[npts::] = np.flip(Rvv[0:npts-1])

Rww = np.empty(npts*2-1)
Rww[0:npts] = data[:,4]
Rww[npts::] = np.flip(Rww[0:npts-1])

Ruv = np.empty(npts*2-1)
Ruv[0:npts] = data[:,5]
Ruv[npts::] = -np.flip(Ruv[0:npts-1])

Ruw = np.empty(npts*2-1)
Ruw[0:npts] = data[:,6]
Ruw[npts::] = np.flip(Ruw[0:npts-1])

Rvw = np.empty(npts*2-1)
Rvw[0:npts] = data[:,7]
Rvw[npts::] = -np.flip(Rvw[0:npts-1])

def stat_interps(u_tau,delta):
    #create interpolation functions from dimensionalized values

    stats = np.empty((3,3,ys.shape[0]))

    stats[0,0,:] = Ruu*u_tau**2
    stats[0,1,:] = Ruv*u_tau**2
    stats[0,2,:] = Ruw*u_tau**2

    stats[1,0,:] = stats[0,1,:]
    stats[1,1,:] = Rvv*u_tau**2
    stats[1,2,:] = Rvw*u_tau**2

    stats[2,0,:] = stats[0,2,:]
    stats[2,1,:] = stats[1,2,:]
    stats[2,2,:] = Rww*u_tau**2

    y = ys*delta
    U = Us*u_tau

    stat = interp1d(y, stats, kind='linear',axis=-1,bounds_error=False,
                    fill_value=(stats[:,:,0],stats[:,:,-1]), assume_sorted=True)

    Ubar = interp1d(y, U, kind='linear', bounds_error=False,
                    fill_value=(U[0],U[-1]), assume_sorted=True)


    return stat,Ubar


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    yplot = np.linspace(0,2,100)

    stat,Ubar = stat_interps(1.0,1.0)

    Rij = stat(yplot)

    Ruu_plot = Rij[0,0,:]
    Rvv_plot = Rij[1,1,:]
    Rww_plot = Rij[2,2,:]

    Ruv_plot = Rij[0,1,:]
    Ruw_plot = Rij[0,2,:]
    Rvw_plot = Rij[1,2,:]

    Uplot = Ubar(yplot)

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax1 = ax[0]
    ax1.set_xlabel(r'$y/ \delta$')
    ax1.set_ylabel(r'$R_{ii}/u_{\tau}^{2}$')
    ax1.plot(yplot,Ruu_plot,label=r'$R_{uu}$')
    ax1.plot(yplot,Rvv_plot,label=r'$R_{vv}$')
    ax1.plot(yplot,Rww_plot,label=r'$R_{ww}$')
    ax1.legend(loc='upper left')
    ax1.set_title('Moser Channel Reynolds Stress')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$R_{ij}/u_{\tau}^{2}$')
    ax2.plot(yplot,Ruv_plot,label='$R_{uv}$',linestyle='--')
    ax2.plot(yplot,Ruw_plot,label='$R_{vw}$',linestyle='--')
    ax2.plot(yplot,Rvw_plot,label='$R_{vw}$',linestyle='--')
    ax2.legend(loc='upper right')

    ax3 = ax[1]
    ax3.set_ylabel(r'$U^{+}$')
    ax3.set_xlabel(r'$y/ \delta$')
    ax3.plot(yplot,Uplot,label='$U^{+}$')

    fig.tight_layout()
    plt.show()
