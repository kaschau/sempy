import numpy as np

"""

Constant, uniform mean flow

"""


def add_profile(domain):
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

    domain.Ubar_interp = lambda y: np.ones(y.shape) * domain.Ublk
