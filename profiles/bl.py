import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf

'''

BOUNDARY LAYER MEAN FLOW PROFILE

Straight forward mean profile construction from theory, see Pope chapter 7.
For outer region blending and wake-law treatment we use

        Revisiting the law of the wake in wall turbulence
        J. Fluid Mech.(2017),vol. 811,pp.421–435
        Cambridge University Press 2016
        doi:10.1017/jfm.2016.788421

See References/Paper/Revisiting-the-law-of-the-wake-in-wall-turbulence_2016.pdf

'''

def add_profile(domain):
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
            Interpolation functions with input y = *dimensional height above the bottom wall*
    '''

    #Viscous sublayer y+=[0,5]
    yplus_vsl = np.linspace(0,5,3)
    us_vsl = yplus_vsl*domain.utau
    ys_vsl = yplus_vsl*domain.viscosity/domain.utau

    #Log-law region y+>30, y+<=3(Re_tau)^(1/2)
    Re_tau = domain.utau*domain.delta/domain.viscosity
    kappa = 0.41
    A = 5.2
    yplus_llr = np.linspace(30, 3*np.sqrt(Re_tau) , 100)
    us_llr = (1.0/kappa * np.log(yplus_llr) + A) * domain.utau
    ys_llr = yplus_llr*domain.viscosity/domain.utau

    #Buffer layer y+=[5,30], we use a CubicSpline to bridge the visc sublayer and log-law region
    yplus_bufl = np.linspace(5,30,10)
    ys_bufl = yplus_bufl*domain.viscosity/domain.utau

    us_bufl = interp1d([ys_vsl[-2],ys_vsl[-1],ys_llr[0],ys_llr[1]],
                       [us_vsl[-2],us_vsl[-1],us_llr[0],us_llr[1]],kind='quadratic')(ys_bufl)

    #Outer region, this is where the wake-law comes into play. See Table 1 from the paper for these values.
    mu = 0.54*domain.delta
    sigma = 0.19*domain.delta
    ys_or = np.linspace(ys_llr[-1], domain.delta , 100)
    yplus_or = ys_or*domain.utau/domain.viscosity

    E = 0.5*(1 + erf((ys_or-mu)/np.sqrt(2*sigma**2)))
    #Make the blending a bit smoother
    s=4; E[0:s] = E[0:s]*np.array([i/s for i in range(s)])
    Up_inf = domain.Ublk/domain.utau
    Up_log = 1.0/kappa * np.log(yplus_or) + A
    uplus_or = Up_inf - (Up_inf - Up_log)*(1-E)
    us_or = uplus_or*domain.utau

    #Put them all together
    ys = np.concatenate((ys_vsl,ys_bufl,ys_llr,ys_or))
    Us = np.concatenate((us_vsl,us_bufl,us_llr,us_or))

    domain.Ubar_interp = interp1d(ys, Us, kind='linear', bounds_error=False,
                                  fill_value=(Us[0],Us[-1]))

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
                            np.linspace(0.05*domain.delta,1.15*domain.delta,100)))

    add_profile(domain)

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
