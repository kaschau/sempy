import numpy as np
from scipy.interpolate import interp1d

'''

CHANNEL FLOW

Mean turbulenct velocity profiles
for fully developed turbulent flow as a function of height above bottom wall from
DNS data of Moser, Kim & Mansour ("DNS of Turbulent Channel Flow up to Re_tau = 590,"
Physics of Fluids, 11: 943-945, 1999).

Data defined for channel flow from y/delta = {0,2} where

        0 == bottom wall
        1 == channel half height
        2 == top wall

The original data is defined from the wall to the channel half height (y/delta=1)
so we take that data and flip it so it is defined everywhere.

'''


try:
    data = np.genfromtxt('./Moser_Channel_ReTau590.csv',delimiter=',',comments='#',skip_header=5)
except:
    data = np.genfromtxt('./profiles/Moser_Channel_ReTau590.csv',delimiter=',',comments='#',skip_header=5)

npts = data.shape[0]

ys = np.empty(npts*2-1)
ys[0:npts] = data[:,0]
ys[npts::] = 2.0 - np.flip(ys[0:npts-1])

Us = np.empty(npts*2-1)
Us[0:npts] = data[:,1]
Us[npts::] = np.flip(Us[0:npts-1])

def add_profile_info(domain):
    ''' Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
      domain : sempy.channel
            Channel object to populate with the sigma interpolator, and sigma mins and maxs

    Returns:
    --------
     None :

        Adds attributes to domain object sich as

        Ubar_interp : scipy.interpolate.1dinterp
            Interpolation functions with input y = *height above the bottom wall*
    '''

    y = ys*domain.delta
    U = Us*domain.utau

    domain.Ubar_interp = interp1d(y, U, kind='linear',axis=0,bounds_error=False,
                                 fill_value=(U[0],U[-1]), assume_sorted=True)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    yplot = np.concatenate((np.linspace(0,0.05,1000),np.linspace(0.05,1.95,100),np.linspace(1.95,2.0,1000)))

    #Create dummy channel
    domain = type('channel',(),{})
    Re = 10e5
    domain.viscosity = 1e-5
    domain.delta = 1.0
    domain.Ublk = Re*domain.viscosity/domain.delta
    domain.utau = domain.Ublk/(5*np.log10(Re))

    add_profile_info(domain)

    Uplot = domain.Ubar_interp(yplot)

    fig, ax = plt.subplots(2,1,figsize=(5,10))

    ax1 = ax[0]
    string = 'Moser Profile Re=10e5, '+r'$u_{\tau}=$'+'{:.2f} '.format(domain.utau)+r'$U_{0}=$'+f'{domain.Ublk}'
    ax1.set_title(f'{string}')
    ax1.set_ylabel(r'$y/ \delta$')
    ax1.set_xlabel(r'$\bar{U}$')
    ax1.plot(Uplot,yplot)

    ax2 = ax[1]
    ax2.set_xlabel(r'$y^{+}$')
    ax2.set_ylabel(r'$u^{+}$')
    yplus = yplot*domain.utau/domain.viscosity
    print(yplus[np.where(yplus<80)])
    print(Uplot[np.where(yplus<80)]/domain.utau)
    ax2.plot(yplus[np.where(yplus<800)],Uplot[np.where(yplus<800)])
    #ax2.set_aspect('equal')

    fig.tight_layout()
    plt.show()
