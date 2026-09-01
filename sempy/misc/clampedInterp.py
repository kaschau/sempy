# -*- coding: utf-8 -*-

import numpy as np


class clampedLinearInterp:
    """
    Linear interpolation along the first axis with end clamping.

    Replacement for the legacy scipy.interpolate.interp1d usage

        interp1d(x, y, kind="linear", axis=0, bounds_error=False,
                 fill_value=(y[0], y[-1]))

    Inputs above the data range return the last data value, inputs below
    return the first (handy for wall values in a channel/BL). Scalar input
    returns a single y-slice, array input returns an array of y-slices.

    Parameters:
    -----------
      x : numpy.array
            1D, ascending independent variable data
      y : numpy.array
            Dependent variable data of shape (len(x), ...) interpolated
            along the first axis
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

    def __call__(self, xi):
        xi = np.asarray(xi, dtype=np.float64)
        scalarInput = xi.ndim == 0
        xi = np.atleast_1d(xi)
        xi = np.clip(xi, self.x[0], self.x[-1])

        idx = np.clip(
            np.searchsorted(self.x, xi, side="right") - 1, 0, self.x.shape[0] - 2
        )
        x0 = self.x[idx]
        x1 = self.x[idx + 1]
        dx = x1 - x0
        # Guard against repeated x data points
        w = np.where(dx == 0.0, 0.0, (xi - x0) / np.where(dx == 0.0, 1.0, dx))
        # Broadcast weights over any trailing dimensions of y
        w = w.reshape(w.shape + (1,) * (self.y.ndim - 1))
        yi = (1.0 - w) * self.y[idx] + w * self.y[idx + 1]

        if scalarInput:
            return yi[0]
        return yi
