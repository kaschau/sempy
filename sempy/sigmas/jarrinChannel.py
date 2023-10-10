# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d

"""

CHANNEL FLOW

Estimates of time and length scales used to construct sigmas pulled from
Figure 6.11 and 7.4 of References/Synthetic-Inflow-Boundary-Conditions-for-the-Numerical-Simulation-of-Turbulence_2008.pdf

Data defined for channel flow from y/delta = {0,2} where 0 == bottom wall and
2 == top wall.

The original data is defined from the wall to the channel half height (y/delta=1)
so we take that data and flip it so it is defined everywhere.

"""

npts = 23

# y/delta data arrays
ys = np.empty(npts * 2 - 1)
ys[0:npts] = np.array(
    [
        0.00000000e00,
        8.51059984e-03,
        1.27659999e-02,
        2.12769993e-02,
        3.06626819e-02,
        4.25530002e-02,
        5.53190000e-02,
        7.23399967e-02,
        9.36169997e-02,
        1.19149998e-01,
        1.44679993e-01,
        1.74470007e-01,
        2.17020005e-01,
        2.59570003e-01,
        3.10640007e-01,
        3.65960002e-01,
        4.29789990e-01,
        5.02129972e-01,
        5.82979977e-01,
        6.68089986e-01,
        7.57449985e-01,
        8.51059973e-01,
        1.00000000e00,
    ]
)
ys[npts::] = 2.0 - np.flip(ys[0 : npts - 1])


# T_ii * u_tau / delta
Tuu = np.empty(ys.shape[0])
Tuu[0:npts] = np.array(
    [
        7.77710080e-02,
        8.16985890e-02,
        8.55177641e-02,
        9.01325867e-02,
        9.62649882e-02,
        1.03042774e-01,
        1.09839179e-01,
        1.18153162e-01,
        1.26509994e-01,
        1.18286796e-01,
        1.02410004e-01,
        7.83130005e-02,
        6.14176169e-02,
        5.18070012e-02,
        4.39065248e-02,
        3.93717214e-02,
        3.61450016e-02,
        3.38928662e-02,
        3.20006236e-02,
        2.99032070e-02,
        2.82403454e-02,
        2.73731723e-02,
        2.70975325e-02,
    ]
)
Tuu[npts::] = np.flip(Tuu[0 : npts - 1])

Tvv = np.empty(ys.shape[0])
Tvv[0:npts] = np.array(
    [
        4.32911664e-02,
        4.48190831e-02,
        4.60610017e-02,
        4.72729988e-02,
        4.84849997e-02,
        4.75981943e-02,
        4.62777987e-02,
        4.44144048e-02,
        4.21184227e-02,
        3.86104360e-02,
        3.46977897e-02,
        3.08449827e-02,
        2.75145359e-02,
        2.57595535e-02,
        2.50007771e-02,
        2.43307594e-02,
        2.39168108e-02,
        2.35234983e-02,
        2.31983084e-02,
        2.29618773e-02,
        2.26160511e-02,
        2.25178655e-02,
        2.24094689e-02,
    ]
)
Tvv[npts::] = np.flip(Tvv[0 : npts - 1])

Tww = np.empty(ys.shape[0])
Tww[0:npts] = np.array(
    [
        3.62598039e-02,
        3.79133970e-02,
        3.90239991e-02,
        3.78049985e-02,
        3.65849994e-02,
        3.50408033e-02,
        3.29269990e-02,
        3.05146296e-02,
        2.81573962e-02,
        2.65855696e-02,
        2.52848100e-02,
        2.42008436e-02,
        2.30626035e-02,
        2.23038271e-02,
        2.16534473e-02,
        2.12198608e-02,
        2.06502397e-02,
        2.03791726e-02,
        2.01623794e-02,
        1.98371895e-02,
        1.97287928e-02,
        1.95119996e-02,
        1.95119996e-02,
    ]
)
Tww[npts::] = np.flip(Tww[0 : npts - 1])

# L_ii
Luu = np.empty(ys.shape[0])
Luu[0:npts] = -0.3415 * ys[0:npts] * (ys[0:npts] - 2.0) + 0.0585
Luu[npts::] = np.flip(Luu[0 : npts - 1])

Lvv = np.empty(ys.shape[0])
Lvv[0:npts] = 0.4050 * ys[0:npts] + 0.0250
Lvv[npts::] = np.flip(Lvv[0 : npts - 1])

Lww = np.empty(ys.shape[0])
Lww[0:npts] = -0.3968 * ys[0:npts] * (ys[0:npts] - 2.0) + 0.0702
Lww[npts::] = np.flip(Lww[0 : npts - 1])


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

    """

    sigmas = np.empty((ys.shape[0], 3, 3))

    sigmas[:, 0, 0] = Tuu * domain.delta / domain.utau * domain.Uo
    sigmas[:, 0, 1] = Luu * domain.delta
    sigmas[:, 0, 2] = Luu * domain.delta

    sigmas[:, 1, 0] = Tvv * domain.delta / domain.utau * domain.Uo
    sigmas[:, 1, 1] = Lvv * domain.delta
    sigmas[:, 1, 2] = Lvv * domain.delta

    sigmas[:, 2, 0] = Tww * domain.delta / domain.utau * domain.Uo
    sigmas[:, 2, 1] = Lww * domain.delta
    sigmas[:, 2, 2] = Lww * domain.delta

    sigmas = sigmas * scaleFactor

    y = ys * domain.delta

    domain.sigmaInterp = interp1d(
        y,
        sigmas,
        kind="slinear",
        axis=0,
        bounds_error=False,
        fill_value=(sigmas[0, :, :], sigmas[-1, :, :]),
    )

    # determine min,max sigmas
    # Here we assume that signal generation at smallest y value is at yplus=1
    testYs = np.linspace(domain.yp1, domain.yHeight - domain.yp1, 200)
    testSigmas = domain.sigmaInterp(testYs)

    domain.sigmaXMin = np.min(testSigmas[:, :, 0])
    domain.sigmaXMax = np.max(testSigmas[:, :, 0])
    domain.sigmaYMin = np.min(testSigmas[:, :, 1])
    domain.sigmaYMax = np.max(testSigmas[:, :, 1])
    domain.sigmaZMin = np.min(testSigmas[:, :, 2])
    domain.sigmaZMax = np.max(testSigmas[:, :, 2])

    domain.vSigmaMin = np.min(np.product(testSigmas, axis=2))
    domain.vSigmaMax = np.max(np.product(testSigmas, axis=2))


if __name__ == "__main__":
    # Create dummy channel
    domain = type("channel", (), {})
    domain.viscosity = 1.0
    domain.utau = 1.0
    domain.delta = 1.0
    domain.yHeight = 2 * domain.delta
    domain.Uo = 1.0
    domain.yp1 = 1e-8
    addSigmas(domain)

    yplot = np.linspace(0, domain.yHeight, 100)
    sigmas = domain.sigmaInterp(yplot)

    tuuPlot = sigmas[:, 0, 0]
    luuPlot = sigmas[:, 0, 1]

    tvvPlot = sigmas[:, 1, 0]
    lvvPlot = sigmas[:, 1, 1]

    twwPlot = sigmas[:, 2, 0]
    lwwPlot = sigmas[:, 2, 1]

    import matplotlib.pyplot as plt
    import shutil

    if shutil.which("latex"):
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams["figure.figsize"] = (6, 4.5)
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 14
    plt.rcParams["lines.linewidth"] = 1.0

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r"$y/ \delta$")
    ax1.set_ylabel(r"$T u_{\tau} / \delta $")
    ax1.plot(yplot, tuuPlot, label=r"$T_{uu}$")
    ax1.plot(yplot, tvvPlot, label=r"$T_{vv}$")
    ax1.plot(yplot, twwPlot, label=r"$T_{ww}$")
    ax1.legend(loc="upper left")
    ax1.set_title("Interpolation Functions for T and L")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$L/ \delta$")
    ax2.plot(yplot, luuPlot, label="$L_{uu}$", linestyle="--")
    ax2.plot(yplot, lvvPlot, label="$L_{vv}$", linestyle="--")
    ax2.plot(yplot, lwwPlot, label="$L_{ww}$", linestyle="--")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()
