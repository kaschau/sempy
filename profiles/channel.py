import numpy as np
from scipy.interpolate import interp1d,splev
from scipy.special import erf

'''

CHANNEL MEAN FLOW PROFILE

Straight forward mean profile construction from theory, see Pope chapter 7.

For outer region modeling and wake-law treatment we use

        Revisiting the law of the wake in wall turbulence
        J. Fluid Mech.(2017),vol. 811,pp.421–435
        Cambridge University Press 2016
        doi:10.1017/jfm.2016.788421

See References/Papers/Revisiting-the-law-of-the-wake-in-wall-turbulence_2016.pdf

There seems to be a pretty strong effect on the log->outer region when different values of
\kappa and A (log-law constants) are used. When the kappa and A values reported for a
channel in the paper above (see Sec 3.1) are used the outer region matches Moser ReTau=590
data very well, but the log layer is off. When the classical values of \kappa=0.41 and A=5.2
are used, both the log layer and outer region match Moser well. Need to test against other
profiles. Above paper reports that outer modeling is fairly insensitive to these log-law
constants.

'''

def add_profile(domain):
    ''' Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
      domain : sempy.channel
            Channel object to populate with the sigma interpolator

    Returns:
    --------
     None :

        Adds attributes to domain

        Ubar_interp : scipy.interpolate.1dinterp
            Interpolation functions with input y = *dimensionsal height above the bottom wall*
    '''

    #Some preamble is needed before constricting the boundary layer profile. We must determine
    # if we have values of kappa and A that in fact lead to a velocity deficit if we extend the
    # the log-layer out to delta. As this model is predicated on that being true. I have found
    # that kappa and A can easily make the log law overshoot Ublk if it is extended into the
    # outer region. It seems kappa is more agreed upon in the literature, so first, we shoot for
    # values of A that yeild a wake deficit
    # similar to the classical wake deficit parameter 2\Pi/\kappa.

    Re_tau = domain.utau*domain.delta/domain.viscosity
    Up_inf = domain.Ublk/domain.utau
    kappa = 0.4
    psi = 0.15
    #Solve for A such that Eq. 2.8 with spec'd value of \Pi is satisfied
    A = Up_inf - 1/kappa*np.log(Re_tau) - psi

    #Construct the profile layer by layer
    #Viscous sublayer y+=[0,5]
    yplus_vsl = np.linspace(0,5,5)
    uplus_vsl = yplus_vsl
    us_vsl = uplus_vsl*domain.utau
    ys_vsl = yplus_vsl*domain.viscosity/domain.utau

    #Log-law region y+>30, y+<=3(Re_tau)^(1/2)
    yplus_llr = np.linspace(30, 3*np.sqrt(Re_tau) , 100)
    uplus_llr = 1.0/kappa * np.log(yplus_llr) + A
    us_llr = uplus_llr*domain.utau
    ys_llr = yplus_llr*domain.viscosity/domain.utau

    #Buffer layer y+=[5,30], we use a BSpline with the intersection of
    # the lines y+=u+ (vsl) and the line from the log-layer to the wall
    # as a control point.
    intersection = [(uplus_llr[0]-1/kappa)/(1-1/(kappa*30))]
    intersection.append(intersection[0])
    cv = np.array([[yplus_vsl[-1],uplus_vsl[-1]],
                   [intersection[0],intersection[1]],
                   [yplus_llr[0],uplus_llr[0]]])
    degree = 2
    count= 3
    kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')

    yplus_bufl,uplus_bufl = np.array(splev(np.linspace(0,(count-degree),30),(kv,cv.T,degree)))
    us_bufl = uplus_bufl*domain.utau
    ys_bufl = yplus_bufl*domain.viscosity/domain.utau

    #Outer region, this is where the wake-law comes into play. See Table 1 from the paper for these values.
    mu = 0.75*domain.delta
    sigma = 0.25*domain.delta
    ys_or = np.linspace(ys_llr[-1], domain.delta , 100)
    yplus_or = ys_or*domain.utau/domain.viscosity

    E = 0.5*(1 + erf((ys_or - mu)/np.sqrt(2*sigma**2)) )
    #Make the blending a bit smoother
    s=4; E[0:s] = E[0:s]*np.array([i/s for i in range(s)])
    E[0] = 0
    Up_log = 1.0/kappa * np.log(yplus_or) + A
    uplus_or = Up_inf - (Up_inf - Up_log)*(1-E)
    us_or = uplus_or*domain.utau

    #Put them all together
    ys = np.concatenate((ys_vsl,ys_bufl,ys_llr,ys_or))
    Us = np.concatenate((us_vsl,us_bufl,us_llr,us_or))

    #Flip to define for entire channel
    ys = np.concatenate((ys, 2.0*domain.delta-np.flip(ys)))
    Us = np.concatenate((Us, np.flip(Us)))

    domain.Ubar_interp = interp1d(ys, Us, kind='linear', bounds_error=False,
                                  fill_value=(Us[0],Us[-1]))

if __name__ == "__main__":

    from pathlib import Path

    #Create dummy channel
    domain = type('channel',(),{})
    Re_tau = 587.19
    domain.viscosity = 1.81e-5
    domain.delta = 0.05
    domain.utau = Re_tau*domain.viscosity/domain.delta
    domain.Ublk = 2.12630000E+01*domain.utau

    ys = np.concatenate((np.linspace(0,0.05*domain.delta,1000),
                         np.linspace(0.05*domain.delta,1.0*domain.delta,100)))

    add_profile(domain)

    Us = domain.Ubar_interp(ys)

    relpath = Path(__file__).parent / "Moser_Channel_ReTau590_Profile.csv"
    data = np.genfromtxt(relpath,delimiter=',',comments='#',skip_header=5)

    yp = data[:,0]
    npts = yp.shape[0]
    MU = data[:,1]

    yd = yp/yp.max()

    import matplotlib.pyplot as plt
    import matplotlib
    if matplotlib.checkdep_usetex(True):
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['figure.figsize'] = (6,4.5)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.size'] = 14
    plt.rcParams['lines.linewidth'] = 1.0

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,10))
    string = r'$Re_{\tau}=590$, '+r'$u_{\tau}=$'+f'{domain.utau:.2f} '+r'$U_{0}=$'+f'{domain.Ublk:.2f}'
    ax1.set_title(f'{string}')
    ax1.set_ylabel(r'$y/ \delta$')
    ax1.set_xlabel(r'$\overline{U}^{+}$')
    ax1.plot(Us/domain.utau,ys/domain.delta, label='SEM')
    ax1.plot(MU,yd, label='Moser')
    ax1.legend()

    ax2.set_xlabel(r'$y^{+}$')
    ax2.set_ylabel(r'$u^{+}$')
    yplus = ys*domain.utau/domain.viscosity
    ax2.plot(yplus[np.where(yplus<100)],Us[np.where(yplus<100)]/domain.utau)
    ax2.plot(yp[np.where(yp<100)],MU[np.where(yp<100)])
    ax2.set_aspect('equal')

    plt.show()
