import numpy as np

"""

ISOTROPIC TURBULENCE FLOW

"""


def rijInterp(y):
    I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    try:
        rij = np.repeat(I[np.newaxis, :, :], y.shape[0], axis=0)
    except IndexError:
        rij = I

    return rij


def addStats(domain):
    """Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
      domain : sempy.channel
            Channel object to populate with the sigma interpolator, and sigma mins and maxs

    Returns:
    --------
     None :

        Adds attributes to domain object sich as

        Rij_interp :
            Interpolation functions with input y = *height above the bottom wall*
            Note that this interpolation function resclaes the data such that it is
            for your channel, not a non dimensionalized channel. These stats and
            profiles are ready to use in your channel and have been
            dimensionalized according to your input values. The shape of the resulting
            sigma array from a call to this interpolate object is:

                 Rij_interp(y) = len(y) x 3 x 3

            Where the last two axes are an array of R_{ij} with the ordering

                    [ [ R_uu   R_uv   R_uw ],
                      [ R_vu   R_vv   R_vw ],
                      [ R_wu   R_wv   R_ww ] ]

    """

    domain.rijInterp = rijInterp
