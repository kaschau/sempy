# -*- coding: utf-8 -*-

import numpy as np

"""

CHANNEL FLOW

Uniform sigmas in all directions for all velocity components. Set by the scale_factor passed to the
function below

"""


def add_sigmas(domain, scale_factor=1.0):
    """Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------

      domain : sempy.channel
            Channel object to populate with the sigma interpolator, and sigma mins and maxs

      scale_factor : float
            Scaling factor for default sigmas, keep 1 unless your doing some sensitivity study

    Returns:
    --------

       None :
            Adds attributes to domain object such as

        sigma_interps
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

    """

    def sigma_interp(y):
        return np.ones((y.shape[0], 3, 3)) * scale_factor

    domain.sigma_interp = sigma_interp

    domain.sigma_x_min = scale_factor
    domain.sigma_x_max = scale_factor
    domain.sigma_y_min = scale_factor
    domain.sigma_y_max = scale_factor
    domain.sigma_z_min = scale_factor
    domain.sigma_z_max = scale_factor

    domain.V_sigma_min = scale_factor ** 3
    domain.V_sigma_max = scale_factor ** 3


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Create dummy channel
    domain = type("channel", (), {})
    domain.ymax = 2
    domain.viscosity = 1.0
    domain.utau = 1.0
    domain.delta = 1.0

    add_sigmas(domain)

    yplot = np.linspace(0, 2, 100)
    sigmas = domain.sigma_interp(yplot)

    sigma_plot = sigmas[:, 0, 0]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r"$y/ \delta$")
    ax1.set_ylabel(r"$\sigma$")
    ax1.plot(yplot, sigma_plot)
    ax1.set_title("Interpolation Functions for Sigma")

    fig.tight_layout()
    plt.show()
