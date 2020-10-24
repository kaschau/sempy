import numpy as np
from scipy.interpolate import interp1d

'''

CHANNEL FLOW

Reynolds Stress Tensor values and mean turbulenct velocity profiles
for fully developed turbulent flow as a function of height above bottom wall from
DNS data of Moser, Kim & Mansour ("DNS of Turbulent Channel Flow up to Re_tau = 590,"
Physics of Fluids, 11: 943-945, 1999).

Data defined for channel flow from y/delta = {0,2} where

        0 == bottom wall
        1 == channel half height
        2 == top wall

The original data is defined from the wall to the channel half height (y/delta=1)
so we take that data and flip it so it is defined everywhere.

Statistics normalized as Rij/u_tau^2 and \bar{U}/u_tau

We flip the sign of Ruv and Rvw in the top half of the channel as "v" is pointing in the other direction
from the top wall's perspective, making the data defined contuniously from 0-->2.

'''


try:
    data = np.genfromtxt('./Moser_Channel_ReTau590.csv',delimiter=',',comments='#',skip_header=5)
except:
    data = np.genfromtxt('./stats/Moser_Channel_ReTau590.csv',delimiter=',',comments='#',skip_header=5)

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

def create_stat_interps(delta, u_tau, ymin=0.0):
    ''' Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
        delta : float
            Channel half height

        u_tau : float
            Friction velocity.

        ymin : float
            Offset coordinate for bottom of channel wall

    Returns:
    --------
        stats_interp,Ubar_interp : scipy.interpolate.1dinterp
            Interpolation functions with input y = *height above the bottom wall*
            Note that this interpolation function resclaes the data such that it is
            for your channel, not a non dimensionalized channel. These stats and
            profiles are ready to use in your channel and have been
            dimensionalized according to your input values. The shape of the resulting
            sigma array from a call to this interpolate object is:

                 stat_interp(y) = len(y) x 3 x 3

            Where the last two axes are an array of R_{ij} with the ordering

                    [ [ R_uu   R_uv   R_uw ],
                      [ R_vu   R_vv   R_vw ],
                      [ R_wu   R_wv   R_ww ] ]

            Another neat property of the interpolator is that the out of bounds values are
            set to the bottom and top wall values so calls above or below those y values
            return the values at the wall.
    '''

    stats = np.empty((ys.shape[0],3,3))

    stats[:,0,0] = Ruu*u_tau**2
    stats[:,0,1] = Ruv*u_tau**2
    stats[:,0,2] = Ruw*u_tau**2

    stats[:,1,0] = stats[:,0,1]
    stats[:,1,1] = Rvv*u_tau**2
    stats[:,1,2] = Rvw*u_tau**2

    stats[:,2,0] = stats[:,0,2]
    stats[:,2,1] = stats[:,1,2]
    stats[:,2,2] = Rww*u_tau**2

    y = ys*delta + ymin
    U = Us*u_tau

    stats_interp = interp1d(y, stats, kind='linear',axis=0,bounds_error=False,
                            fill_value=(stats[0,:,:],stats[-1,:,:]), assume_sorted=True)

    Ubar_interp = interp1d(y, U, kind='linear', bounds_error=False,
                           fill_value=(U[0],U[-1]), assume_sorted=True)

    return stats_interp,Ubar_interp


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    yplot = np.linspace(0,2,100)

    stat,Ubar = create_stat_interps(1.0,1.0)

    Rij = stat(yplot)

    Ruu_plot = Rij[:,0,0]
    Rvv_plot = Rij[:,1,1]
    Rww_plot = Rij[:,2,2]

    Ruv_plot = Rij[:,0,1]
    Ruw_plot = Rij[:,0,2]
    Rvw_plot = Rij[:,1,2]

    Uplot = Ubar(yplot)

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax1 = ax[0]
    ax1.set_ylabel(r'$y/ \delta$')
    ax1.set_xlabel(r'$R_{ii}/u_{\tau}^{2}$')
    ax1.plot(Ruu_plot,yplot,label=r'$R_{uu}$')
    ax1.plot(Rvv_plot,yplot,label=r'$R_{vv}$')
    ax1.plot(Rww_plot,yplot,label=r'$R_{ww}$')
    ax1.legend()
    ax1.set_title('Moser Channel Reynolds Stress')

    ax2 = ax1.twiny()
    ax2.set_xlabel(r'$R_{ij}/u_{\tau}^{2}$')
    ax2.plot(Ruv_plot,yplot,label='$R_{uv}$',linestyle='--')
    ax2.plot(Ruw_plot,yplot,label='$R_{vw}$',linestyle='--')
    ax2.plot(Rvw_plot,yplot,label='$R_{vw}$',linestyle='--')
    ax2.legend()

    ax3 = ax[1]
    ax3.set_title(r'Moser $\bar{U}$')
    ax3.set_xlabel(r'$U^{+}$')
    ax3.set_ylabel(r'$y/ \delta$')
    ax3.plot(Uplot,yplot,label='$U^{+}$')

    fig.tight_layout()
    plt.show()
