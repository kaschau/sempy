# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d

'''

BOUNDARY LAYER FLOW

Linear estimates of time and length scales used to construct sigmas

'''

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
    y = np.array([domain.yp1,domain.delta])
    sigmas = np.empty((y.shape[0],3,3))

    sigmas[0,:,:] = domain.yp1
    sigmas[1,:,:] = domain.delta

    sigmas = sigmas*scale_factor

    domain.sigma_interp = interp1d(y, sigmas, kind='slinear',axis=0,bounds_error=False,
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
    domain = type('bl',(),{})
    domain.y_height = 1.2
    domain.viscosity = 1.0
    domain.utau = 1.0
    domain.delta = 1.0
    domain.yp1 = 1e-5
    add_sigmas(domain)

    yplot = np.linspace(0,1.2,100)
    sigmas = domain.sigma_interp(yplot)

    sigma_plot = sigmas[:,0,0]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'$y/ \delta$')
    ax1.set_ylabel(r'$\sigma / \delta$')
    ax1.plot(yplot,sigma_plot)
    ax1.set_title('Interpolation Functions for Linear Sigma BL')

    fig.tight_layout()
    plt.show()
