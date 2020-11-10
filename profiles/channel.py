import numpy as np
from scipy.interpolate import interp1d,CubicSpline

'''

CHANNEL MEAN FLOW PROFILE

Straight forward mean profile construction from theory, see Pope chapter 7.

Data defined for channel flow from y/delta = {0,2} where

        0 == bottom wall
        1 == channel half height (delta)
        2 == top wall

'''

def add_profile_info(domain):
    ''' Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
      domain : sempy.channel
            Channel object to populate with the sigma interpolator, and sigma mins and maxs

    Returns:
    --------
     None :

        Adds attributes to domain

        Ubar_interp : scipy.interpolate.1dinterp
            Interpolation functions with input y = *height above the bottom wall*
            Note that this interpolation function resclaes the data such that it is
            for your channel, not a non dimensionalized channel. These stats and
            profiles are ready to use in your channel and have been
            dimensionalized according to your input values.
    '''

    #Viscous sublayer y+=[0,5]
    yplus_vsl = np.linspace(0,5,3)
    us_vsl = yplus_vsl*domain.utau
    ys_vsl = yplus_vsl*domain.viscosity/domain.utau

    #Log-law refion y+>30, y/delta<0.3
    ys_llr = np.linspace(30*domain.viscosity/domain.utau, 0.3*domain.delta, 100)
    yplus_llr = ys_llr*domain.utau/domain.viscosity
    us_llr = (1.0/0.41 * np.log(yplus_llr) + 5.2) * domain.utau

    #Buffer layer y+=[5,30], we use a CubicSpline to bridge the visc sublayer and log-law region
    yplus_bufl = np.linspace(5,30,10)
    ys_bufl = yplus_bufl*domain.viscosity/domain.utau

    bcs=((1,np.diff(us_vsl[-2::])[0]/np.diff(ys_vsl[-2::])[0]),
         (1,np.diff(us_llr[0:2])[0]/np.diff(ys_llr[0:2])[0]))

    us_bufl = CubicSpline([ys_vsl[-1],ys_llr[0]],
                          [us_vsl[-1],us_llr[0]],bc_type=bcs)(ys_bufl)

    #Outer region, again using a Spline to meet the specified Ublk
    ys_or = np.linspace(0.3*domain.delta,domain.delta, 100)

    bcs=((1,np.diff(us_llr[-2::])[0]/np.diff(ys_llr[-2::])[0]),(1,0.0))

    us_or = CubicSpline([ys_llr[-1],ys_or[-1]],
                        [us_llr[-1],domain.Ublk],bc_type=bcs)(ys_or)

    #Put them all together
    ys = np.concatenate((ys_vsl,ys_bufl,ys_llr,ys_or))
    Us = np.concatenate((us_vsl,us_bufl,us_llr,us_or))

    #Flip to define for entire channel
    ys = np.concatenate((ys, 2.0*domain.delta-np.flip(ys)))
    Us = np.concatenate((Us, np.flip(Us)))

    domain.Ubar_interp = interp1d(ys, Us, kind='linear', bounds_error=False,
                                  fill_value=(Us[0],Us[-1]), assume_sorted=True)

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
    string = 'Re=10e5, '+r'$u_{\tau}=$'+'{:.2f} '.format(domain.utau)+r'$U_{0}=$'+f'{domain.Ublk}'
    ax1.set_title(f'{string}')
    ax1.set_ylabel(r'$y/ \delta$')
    ax1.set_xlabel(r'$\bar{U}$')
    ax1.plot(Uplot,yplot/domain.delta)

    ax2 = ax[1]
    ax2.set_xlabel(r'$y^{+}$')
    ax2.set_ylabel(r'$u^{+}$')
    yplus = yplot*domain.utau/domain.viscosity
    ax2.plot(yplus[np.where(yplus<80)],Uplot[np.where(yplus<80)]/domain.utau)
    ax2.set_aspect('equal')

    fig.tight_layout()
    plt.show()
