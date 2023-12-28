# -*- coding: utf-8 -*-

import numpy as np

"""

CHANNEL FLOW

Uniform sigmas in all directions for all velocity components. Set by the scale_factor passed to the
function below

"""


def addSigmas(domain, scaleFactor=1.0):
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

            Where the last two axes are an array of sigma_{ij} with the ordering

                    [ [ sigma_ux   sigma_uy   sigma_uz ],
                      [ sigma_vx   sigma_vy   sigma_vz ],
                      [ sigma_wx   sigma_wy   sigma_wz ] ]

            Another neat property of the interpolator is that the out of bounds values are
            set to the bottom and top wall values so calls above or below those y values
            return the values at the wall.

    """

    def sigmaInterp(y):
        return np.ones((y.shape[0], 3, 3)) * scaleFactor

    domain.sigmaInterp = sigmaInterp

    domain.sigmaXMin = scaleFactor
    domain.sigmaXMax = scaleFactor
    domain.sigmaYMin = scaleFactor
    domain.sigmaYMax = scaleFactor
    domain.sigmaZMin = scaleFactor
    domain.sigmaZMax = scaleFactor

    domain.vSigmaMin = scaleFactor**3
    domain.vSigmaMax = scaleFactor**3


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create dummy channel
    domain = type("channel", (), {})
    domain.ymax = 2
    domain.viscosity = 1.0
    domain.utau = 1.0
    domain.delta = 1.0

    addSigmas(domain)

    yplot = np.linspace(0, 2, 100)
    sigmas = domain.sigmaInterp(yplot)

    sigma_plot = sigmas[:, 0, 0]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r"$y/ \delta$")
    ax1.set_ylabel(r"$\sigma$")
    ax1.plot(yplot, sigma_plot)
    ax1.set_title("Interpolation Functions for Sigma")

    fig.tight_layout()
    plt.show()
