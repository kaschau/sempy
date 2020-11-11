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

yp = data[:,0]
yd = yp/yp[-1]
Us = data[:,1]

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


    yp_trans = 100
    overlap = np.where(yp > 100)

    #No idea if this is general enough
    transition = np.exp(-np.exp(-7.5*np.linspace(0,1,overlap[0].shape[0])+2.0))

    y = yp*domain.viscosity/domain.utau
    y[overlap] = yp[overlap]*domain.viscosity/domain.utau*np.flip(transition) + yd[overlap]*domain.delta*transition

    U = Us*domain.utau
    U[overlap] = U[overlap] + (domain.Ublk - U[overlap])*transition

    y = np.concatenate((y,2.0*domain.delta-np.flip(y)))
    U = np.concatenate((U,np.flip(U)))

    domain.Ubar_interp = interp1d(y, U, kind='linear',axis=0,bounds_error=False,
                                 fill_value=(U[0],U[-1]), assume_sorted=True)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    #Create dummy channel
    domain = type('channel',(),{})
    Re = 10e5
    domain.viscosity = 1e-5
    domain.delta = 1.0
    domain.Ublk = Re*domain.viscosity/domain.delta
    domain.utau = domain.Ublk/(5*np.log10(Re))

    yplot = np.concatenate((np.linspace(0,0.05*domain.delta,1000),
                            np.linspace(0.05*domain.delta,1.95*domain.delta,100),
                            np.linspace(1.95*domain.delta,2.0*domain.delta,1000)))

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
    uplus = Uplot/domain.utau
    ax2.plot(yplus[np.where(yplus<80)],uplus[np.where(yplus<80)])
#    ax2.set_aspect('equal')

    fig.tight_layout()
    plt.show()
