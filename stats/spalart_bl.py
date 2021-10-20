import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

"""

CHANNEL FLOW

Reynolds Stress Tensor values and mean turbulenct velocity profiles
for fully developed turbulent flow as a function of height above bottom wall from
DNS data of Moser, Kim & Mansour ("DNS of Turbulent Channel Flow up to Re_tau = 590,"
Physics of Fluids, 11: 943-945, 1999).

Data defined for channel flow from y/delta = {0,2} where

        0 == bottom wall
        1 == channel half height
        2 == top wall

The original data is defined from the wall to the channel half height (y/delta=1)
so we take that data and flip it so it is defined everywhere.

Statistics normalized as Rij/u_tau^2 and \bar{U}/u_tau

We flip the sign of Ruv and Rvw in the top half of the channel as "v" is pointing in the other direction
from the top wall's perspective, making the data defined contuniously from 0-->2.

"""


relpath = Path(__file__).parent / "Spalart_BL_ReTau547.csv"
data = np.genfromtxt(relpath, delimiter=",", comments="#", skip_header=1)

ys = data[:, 0]

Ruu = data[:, 1]

Rvv = data[:, 2]

Rww = data[:, 3]

Ruv = data[:, 4]

Ruw = data[:, 5]

Rvw = data[:, 6]


def add_stats(domain):
    """Function that returns a 1d interpolation object creted from the data above.

    Parameters:
    -----------
      domain : sempy.channel
            Channel object to populate with the sigma interpolator, and sigma mins and maxs

    Returns:
    --------
     None :

        Adds attributes to domain object sich as

        stats_interp,Ubar_interp : scipy.interpolate.1dinterp
            Interpolation functions with input y = *height above the bottom wall*
            Note that this interpolation function resclaes the data such that it is
            for your channel, not a non dimensionalized channel. These stats and
            profiles are ready to use in your channel and have been
            dimensionalized according to your input values. The shape of the resulting
            sigma array from a call to this interpolate object is:

                 stat_interp(y) = len(y) x 3 x 3

            Where the last two axes are an array of R_{ij} with the ordering

                    [ [ R_uu   R_uv   R_uw ],
                      [ R_vu   R_vv   R_vw ],
                      [ R_wu   R_wv   R_ww ] ]

            Another neat property of the interpolator is that the out of bounds values are
            set to the bottom and top wall values so calls above or below those y values
            return the values at the wall.
    """

    stats = np.empty((ys.shape[0], 3, 3))

    stats[:, 0, 0] = Ruu * domain.utau ** 2
    stats[:, 0, 1] = Ruv * domain.utau ** 2
    stats[:, 0, 2] = Ruw * domain.utau ** 2

    stats[:, 1, 0] = stats[:, 0, 1]
    stats[:, 1, 1] = Rvv * domain.utau ** 2
    stats[:, 1, 2] = Rvw * domain.utau ** 2

    stats[:, 2, 0] = stats[:, 0, 2]
    stats[:, 2, 1] = stats[:, 1, 2]
    stats[:, 2, 2] = Rww * domain.utau ** 2

    y = ys * domain.delta

    domain.Rij_interp = interp1d(
        y,
        stats,
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value=(stats[0, :, :], stats[-1, :, :]),
    )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    yplot = np.linspace(0, 2, 100)

    # Create dummy bl
    domain = type("bl", (), {})
    domain.y_height = 1.2
    domain.viscosity = 1.0
    domain.utau = 1.0
    domain.delta = 1.0
    add_stats(domain)

    Rij = domain.Rij_interp(yplot)

    Ruu_plot = Rij[:, 0, 0]
    Rvv_plot = Rij[:, 1, 1]
    Rww_plot = Rij[:, 2, 2]

    Ruv_plot = Rij[:, 0, 1]
    Ruw_plot = Rij[:, 0, 2]
    Rvw_plot = Rij[:, 1, 2]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = ax[0]
    ax1.set_ylabel(r"$y/ \delta$")
    ax1.set_xlabel(r"$R_{ii}/u_{\tau}^{2}$")
    ax1.plot(Ruu_plot, yplot, label=r"$R_{uu}$")
    ax1.plot(Rvv_plot, yplot, label=r"$R_{vv}$")
    ax1.plot(Rww_plot, yplot, label=r"$R_{ww}$")
    ax1.legend()
    ax1.set_title("Spalart BL Reynolds Stress")

    ax2 = ax[1]
    ax2.set_xlabel(r"$R_{ij}/u_{\tau}^{2}$")
    ax2.plot(Ruv_plot, yplot, label="$R_{uv}$", linestyle="--")
    ax2.plot(Ruw_plot, yplot, label="$R_{vw}$", linestyle="--")
    ax2.plot(Rvw_plot, yplot, label="$R_{vw}$", linestyle="--")
    ax2.legend()

    fig.tight_layout()
    plt.show()
