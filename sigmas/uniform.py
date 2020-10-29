# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d

'''

CHANNEL FLOW

Estimates of time and length scales used to construct sigmas pulled from
Jarrin, 2008 ?Figure 8.8(a)?

Data defined for channel flow from y/delta = {0,2} where 0 == bottom wall and
2 == top wall.

The original data is defined from the wall to the channel half height (y/delta=1)
so we take that data and flip it so it is defined everywhere.

'''

def create_sigma_interps(delta=None , utau=None, ymin=0.0, r=1.0):
    ''' Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------

        ymin : float
            Offset coordinate for bottom of channel wall

           r : float
            Uniform eddy size for all velocity/direction pairs

    Returns:
    --------
        sigma_interps : scipy.interpolate.1dinterp
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

    def sigma_interps(y):
        return np.ones((y.shape[0],3,3))*r

    return sigma_interps


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    yplot = np.linspace(0,2,100)
    sigmas = create_sigma_interps(1.0,1.0)(yplot)

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
