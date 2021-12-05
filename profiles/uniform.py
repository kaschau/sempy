import numpy as np

"""

Constant, uniform mean flow

"""


def addProfile(domain):
    """Function that returns a callable object creted from the data above.

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
    """

    domain.ubarInterp = lambda y: np.ones(y.shape) * domain.Ublk
