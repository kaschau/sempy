import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

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

relpath = Path(__file__).parent / "Moser_Channel_ReTau590.csv"
data = np.genfromtxt(relpath,delimiter=',',comments='#',skip_header=5)

yp = data[:,0]
yd = yp/yp.max()

npts = yp.shape[0]

Ruu = np.empty(npts*2-1)
Ruu[0:npts] = data[:,1]
Ruu[npts::] = np.flip(Ruu[0:npts-1])

Rvv = np.empty(npts*2-1)
Rvv[0:npts] = data[:,2]
Rvv[npts::] = np.flip(Rvv[0:npts-1])

Rww = np.empty(npts*2-1)
Rww[0:npts] = data[:,3]
Rww[npts::] = np.flip(Rww[0:npts-1])

Ruv = np.empty(npts*2-1)
Ruv[0:npts] = data[:,4]
Ruv[npts::] = -np.flip(Ruv[0:npts-1])

Ruw = np.empty(npts*2-1)
Ruw[0:npts] = data[:,5]
Ruw[npts::] = np.flip(Ruw[0:npts-1])

Rvw = np.empty(npts*2-1)
Rvw[0:npts] = data[:,6]
Rvw[npts::] = -np.flip(Rvw[0:npts-1])

def add_stats(domain):
    ''' Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
      domain : sempy.channel
            Channel object to populate with the sigma interpolator, and sigma mins and maxs

    Returns:
    --------
     None :

        Adds attributes to domain object sich as

        stats_interp : scipy.interpolate.1dinterp
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

    Re_tau = domain.utau*domain.delta/domain.viscosity
    yp_trans = 3*np.sqrt(Re_tau)
    overlap = np.where(yp > yp_trans)

    #No idea if this is general enough
    transition = np.exp(-np.exp(-7.5*np.linspace(0,1,overlap[0].shape[0])+2.0))

    y = yp*domain.viscosity/domain.utau
    y[overlap] = yp[overlap]*domain.viscosity/domain.utau*(1.0-transition) + yd[overlap]*domain.delta*transition
    y = np.concatenate((y,2.0*domain.delta-np.flip(y[1::])))

    stats = np.empty((y.shape[0],3,3))

    stats[:,0,0] = Ruu*domain.utau**2
    stats[:,0,1] = Ruv*domain.utau**2
    stats[:,0,2] = Ruw*domain.utau**2

    stats[:,1,0] = stats[:,0,1]
    stats[:,1,1] = Rvv*domain.utau**2
    stats[:,1,2] = Rvw*domain.utau**2

    stats[:,2,0] = stats[:,0,2]
    stats[:,2,1] = stats[:,1,2]
    stats[:,2,2] = Rww*domain.utau**2

    domain.Rij_interp = interp1d(y, stats, kind='linear',axis=0,bounds_error=False,
                                 fill_value=(stats[0,:,:],stats[-1,:,:]))

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    #Create dummy channel
    domain = type('channel',(),{})
    Re_tau = 587.19
    domain.viscosity = 1.81e-5
    domain.delta = 0.05
    domain.utau = Re_tau*domain.viscosity/domain.delta
    domain.Ublk = 2.12630000E+01*domain.utau

    yplot = np.concatenate((np.linspace(0,0.05*domain.delta,1000)[0:-1],
                            np.linspace(0.05*domain.delta,1.95*domain.delta,500),
                            np.linspace(1.95*domain.delta,2.0*domain.delta,1000)[1::]))

    add_stats(domain)

    Rij = domain.Rij_interp(yplot)

    Ruu_plot = Rij[:,0,0]/domain.utau**2
    Rvv_plot = Rij[:,1,1]/domain.utau**2
    Rww_plot = Rij[:,2,2]/domain.utau**2

    Ruv_plot = Rij[:,0,1]/domain.utau**2
    Ruw_plot = Rij[:,0,2]/domain.utau**2
    Rvw_plot = Rij[:,1,2]/domain.utau**2

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax1 = ax[0]
    ax1.set_ylabel(r'$y/ \delta$')
    ax1.set_xlabel(r'$R_{ii}/u_{\tau}^{2}$')
    ax1.plot(Ruu_plot,yplot,label=r'$uu$')
    ax1.plot(Rvv_plot,yplot,label=r'$vv$')
    ax1.plot(Rww_plot,yplot,label=r'$ww$')
    ax1.legend()
    ax1.set_title('Moser Channel Reynolds Stress')

    ax2 = ax1.twiny()
    ax2.set_xlabel(r'$R_{ij}/u_{\tau}^{2}$')
    ax2.plot(Ruv_plot,yplot,label='$uv$',linestyle='--')
    ax2.plot(Ruw_plot,yplot,label='$vw$',linestyle='--')
    ax2.plot(Rvw_plot,yplot,label='$vw$',linestyle='--')
    ax2.legend()

    ax3 = ax[1]
    yplus = yplot*domain.utau/domain.viscosity
    string = 'Moser Profile, '+r'$u_{\tau}=$'+'{:.2f} '.format(domain.utau)+r'$U_{0}=$'+f'{domain.Ublk}'
    ax3.set_title(f'{string}')
    ax3.set_xlabel(r'$y^{+}$')
    ax3.set_ylabel(r'$R_{ii}/u_{\tau}^{2}$')
    ax3.plot(yplus[np.where((yplus<100))],Ruu_plot[np.where((yplus<100))],label=r'$uu$')
    ax3.plot(yplus[np.where((yplus<100))],Rvv_plot[np.where((yplus<100))],label=r'$vv$')
    ax3.plot(yplus[np.where((yplus<100))],Rww_plot[np.where((yplus<100))],label=r'$ww$')
    ax3.legend()

    fig.tight_layout()
    plt.show()
