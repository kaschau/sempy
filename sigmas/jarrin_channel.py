# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d

'''

CHANNEL FLOW

Estimates of time and length scales used to construct sigmas pulled from
Figure 6.11 and 7.4 of References/Synthetic-Inflow-Boundary-Conditions-for-the-Numerical-Simulation-of-Turbulence_2008.pdf

Data defined for channel flow from y/delta = {0,2} where 0 == bottom wall and
2 == top wall.

The original data is defined from the wall to the channel half height (y/delta=1)
so we take that data and flip it so it is defined everywhere.

'''

npts = 23

# y/delta data arrays
ys = np.empty(npts*2-1)
ys[0:npts] = np.array([0.00000000E+00, 8.51059984E-03, 1.27659999E-02, 2.12769993E-02,
                       3.06626819E-02, 4.25530002E-02, 5.53190000E-02, 7.23399967E-02,
                       9.36169997E-02, 1.19149998E-01, 1.44679993E-01, 1.74470007E-01,
                       2.17020005E-01, 2.59570003E-01, 3.10640007E-01, 3.65960002E-01,
                       4.29789990E-01, 5.02129972E-01, 5.82979977E-01, 6.68089986E-01,
                       7.57449985E-01, 8.51059973E-01, 1.00000000E+00])
ys[npts::] = 2.0 - np.flip(ys[0:npts-1])


# T_ii * u_tau / delta
Tuu = np.empty(ys.shape[0])
Tuu[0:npts] = np.array([7.77710080E-02, 8.16985890E-02, 8.55177641E-02, 9.01325867E-02,
                        9.62649882E-02, 1.03042774E-01, 1.09839179E-01, 1.18153162E-01,
                        1.26509994E-01, 1.18286796E-01, 1.02410004E-01, 7.83130005E-02,
                        6.14176169E-02, 5.18070012E-02, 4.39065248E-02, 3.93717214E-02,
                        3.61450016E-02, 3.38928662E-02, 3.20006236E-02, 2.99032070E-02,
                        2.82403454E-02, 2.73731723E-02, 2.70975325E-02])
Tuu[npts::] = np.flip(Tuu[0:npts-1])

Tvv = np.empty(ys.shape[0])
Tvv[0:npts] = np.array([4.32911664E-02, 4.48190831E-02, 4.60610017E-02, 4.72729988E-02,
                        4.84849997E-02, 4.75981943E-02, 4.62777987E-02, 4.44144048E-02,
                        4.21184227E-02, 3.86104360E-02, 3.46977897E-02, 3.08449827E-02,
                        2.75145359E-02, 2.57595535E-02, 2.50007771E-02, 2.43307594E-02,
                        2.39168108E-02, 2.35234983E-02, 2.31983084E-02, 2.29618773E-02,
                        2.26160511E-02, 2.25178655E-02, 2.24094689E-02])
Tvv[npts::] = np.flip(Tvv[0:npts-1])

Tww = np.empty(ys.shape[0])
Tww[0:npts] = np.array([3.62598039E-02, 3.79133970E-02, 3.90239991E-02, 3.78049985E-02,
                        3.65849994E-02, 3.50408033E-02, 3.29269990E-02, 3.05146296E-02,
                        2.81573962E-02, 2.65855696E-02, 2.52848100E-02, 2.42008436E-02,
                        2.30626035E-02, 2.23038271E-02, 2.16534473E-02, 2.12198608E-02,
                        2.06502397E-02, 2.03791726E-02, 2.01623794E-02, 1.98371895E-02,
                        1.97287928E-02, 1.95119996E-02, 1.95119996E-02])
Tww[npts::] = np.flip(Tww[0:npts-1])

# L_ii
Luu = np.empty(ys.shape[0])
Luu[0:npts] = - 0.3415*ys[0:npts]*(ys[0:npts] - 2.0) + 0.0585
Luu[npts::] = np.flip(Luu[0:npts-1])

Lvv = np.empty(ys.shape[0])
Lvv[0:npts] =   0.4050*ys[0:npts]                    + 0.0250
Lvv[npts::] = np.flip(Lvv[0:npts-1])

Lww = np.empty(ys.shape[0])
Lww[0:npts] = - 0.3968*ys[0:npts]*(ys[0:npts] - 2.0) + 0.0702
Lww[npts::] = np.flip(Lww[0:npts-1])

def add_sigmas(domain, scale_factor=1.0):
    ''' Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
      domain : sempy.channel
            Channel object to populate with the sigma interpolator, and sigma mins and maxs

scale_factor : float
            Scaling factor for default sigmas, keep 1 unless your doing some sensitivity study

    Returns:
    --------
        sigma_interp : scipy.interpolate.1dinterp
            Interpolation function with input y = *height above the bottom wall*
            Note that this interpolation function resclaes the data such that it is
            for your channel, not a non dimensionalized channel. Same with the sigmas,
            these length scales are ready to use in your channel and have been
            dimensionalized according to your input values. The shape of the resulting
            sigma array from a call to this interpolate object is:

                 sigma_interp(y) = len(y) x 3 x 3

            Where the last two axes are an array of \sigma_{ij} with the ordering

                    [ [ sigma_ux   sigma_uy   sigma_uz ],
                      [ sigma_vx   sigma_vy   sigma_vz ],
                      [ sigma_wx   sigma_wy   sigma_wz ] ]

            Another neat property of the interpolator is that the out of bounds values are
            set to the bottom and top wall values so calls above or below those y values
            return the values at the wall.

    '''

    sigmas = np.empty((ys.shape[0],3,3))

    sigmas[:,0,0] = Tuu*domain.delta/domain.utau
    sigmas[:,0,1] = Luu*domain.delta
    sigmas[:,0,2] = Luu*domain.delta

    sigmas[:,1,0] = Tvv*domain.delta/domain.utau
    sigmas[:,1,1] = Lvv*domain.delta
    sigmas[:,1,2] = Lvv*domain.delta

    sigmas[:,2,0] = Tww*domain.delta/domain.utau
    sigmas[:,2,1] = Lww*domain.delta
    sigmas[:,2,2] = Lww*domain.delta

    sigmas = sigmas*scale_factor

    sigmas[:,0,0] = np.clip(sigmas[:,0,0], sigmas[:,0,1].min(), None)
    sigmas[:,1,0] = np.clip(sigmas[:,1,0], sigmas[:,1,1].min(), None)
    sigmas[:,2,0] = np.clip(sigmas[:,2,0], sigmas[:,2,1].min(), None)
    y = ys*domain.delta

    domain.sigma_interp = interp1d(y, sigmas, kind='linear',axis=0,bounds_error=False,
                                   fill_value=(sigmas[0,:,:],sigmas[-1,:,:]))

    #determine min,max sigmas
    # Here we assume that signal generation at smallest y value is at yplus=1
    test_ys = np.linspace(domain.yp1,domain.y_height-domain.yp1,200)
    test_sigmas = domain.sigma_interp(test_ys)

    domain.sigma_x_min = np.min(test_sigmas[:,:,0])
    domain.sigma_x_max = np.max(test_sigmas[:,:,0])
    domain.sigma_y_min = np.min(test_sigmas[:,:,1])
    domain.sigma_y_max = np.max(test_sigmas[:,:,1])
    domain.sigma_z_min = np.min(test_sigmas[:,:,2])
    domain.sigma_z_max = np.max(test_sigmas[:,:,2])

    domain.V_sigma_min = np.min(np.product(test_sigmas,axis=1))
    domain.V_sigma_max = np.max(np.product(test_sigmas,axis=1))

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    #Create dummy channel
    domain = type('channel',(),{})
    domain.ymax = 2
    domain.viscosity = 1.0
    domain.utau = 1.0
    domain.delta = 1.0
    domain.yp1 = 1e-5
    add_sigmas(domain)

    yplot = np.linspace(0,2,100)
    sigmas = domain.sigma_interp(yplot)

    Tuu_plot = sigmas[:,0,0]
    Luu_plot = sigmas[:,0,1]

    Tvv_plot = sigmas[:,1,0]
    Lvv_plot = sigmas[:,1,1]

    Tww_plot = sigmas[:,2,0]
    Lww_plot = sigmas[:,2,1]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'$y/ \delta$')
    ax1.set_ylabel(r'$T u_{\tau} / \delta $')
    ax1.plot(yplot,Tuu_plot,label=r'$T_{uu}$')
    ax1.plot(yplot,Tvv_plot,label=r'$T_{vv}$')
    ax1.plot(yplot,Tww_plot,label=r'$T_{ww}$')
    ax1.legend(loc='upper left')
    ax1.set_title('Interpolation Functions for T and L')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$L/ \delta$')
    ax2.plot(yplot,Luu_plot,label='$L_{uu}$',linestyle='--')
    ax2.plot(yplot,Lvv_plot,label='$L_{vv}$',linestyle='--')
    ax2.plot(yplot,Lww_plot,label='$L_{ww}$',linestyle='--')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()
