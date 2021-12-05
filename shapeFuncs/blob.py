import numpy as np
from numba import njit


@njit(fastmath=True)
def blob(dists, sigmas):
    """
    Produce an eddy with single volume, and 9 characteristic length scales.

    NOTE: distances in dists are not absolute valued.
    They can be positive and negative. Also not all the eddys provided will contribute
    to all velocity components. So you need to zero out manually the ones that dont.

    Parameters:
    -----------
      dist   : numpy.array
            Array of shape(N,3) where N is the number of eddys being considered
      sigmas : numpy.array
            Array of shape(N,3,3) where N is the number of eddys being considered,
            same shape as the usual sigma array...

                    [ [ sigma_ux,  sigma_uy,  sigma_uz ],
                      [ sigma_vx,  sigma_vy,  sigma_vz ],
                      [ sigma_wx,  sigma_wy,  sigma_wz ] ]

    Returns:
    --------
        fx : numpy.arrray
            Array of shape(N,3) of the fluctuation contributions for each of the N
            eddys for the 3 velocity fluctuation components.
    """

    fxx = np.ones((dists.shape[0], 3))
    for i, (d, s) in enumerate(zip(dists, sigmas)):
        if np.max(np.abs(d)) > np.max(s):
            fxx[i, :] = 0.0
            continue
        fx = np.empty((3, 3))

        # Loop over each x,y,z direction, find the largest length scale in that direction,
        # compute the "big" eddy, and then the other two small eddys
        for xyz in range(3):
            if np.abs(d[xyz]) > np.max(s[:, xyz]):
                fx[:, xyz] = 0.0
                break

            maxComponent = np.argmax(s[:, xyz])
            bigEddy = np.cos(np.pi * d[xyz] / (2.0 * s[maxComponent, xyz]))

            fx[maxComponent, xyz] = bigEddy

            smaller_components = [0, 1, 2]
            smaller_components.remove(maxComponent)
            for smlr in smaller_components:
                small_eddy = bigEddy * np.cos(np.pi * d[xyz] / (2.0 * s[smlr, xyz]))
                fx[smlr, xyz] = small_eddy

        for k in range(3):
            fxx[i, k] = np.prod(fx[k])

    # Total f(x) from each contributing eddy
    return fxx


if __name__ == "__main__":

    nsig = 201
    uxLengthScale = 1.0
    uyLengthScale = 0.75
    vxLengthScale = 0.5
    vyLengthScale = 0.25
    wxLengthScale = 0.25
    wyLengthScale = 0.25

    line = np.linspace(-uxLengthScale * 1.5, uxLengthScale * 1.5, nsig)

    dists = np.zeros((nsig, 3))
    dists[:, 0] = line

    sigmas = np.ones((nsig, 3, 3))
    sigmas[:, 0, 0] = uxLengthScale
    sigmas[:, 1, 0] = vxLengthScale
    sigmas[:, 2, 0] = wxLengthScale

    sigs = blob(dists, sigmas)

    import matplotlib.pyplot as plt
    import matplotlib

    if matplotlib.checkdep_usetex(True):
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams["figure.figsize"] = (6, 4.5)
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 14
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["image.cmap"] = "seismic"

    fig, ax = plt.subplots()
    lo = ax.plot(line, sigs)
    ax.set_title(r"$f(|x-\sigma_{i}|)$")
    ax.set_xlabel(r"$x$")
    plt.legend(
        lo,
        (
            r"$\sigma_{u}$=" + f"{uxLengthScale}",
            r"$\sigma_{v}=$" + f"{vxLengthScale}",
            r"$\sigma_{w}=$" + f"{wxLengthScale}",
        ),
    )
    plt.grid(linestyle="--")
    plt.show()

    ##############################################################
    # Optional 2D plots
    ##############################################################
    plot_2D = True
    if plot_2D:
        X, Y = np.meshgrid(line, line)
        npts = X.ravel().shape[0]

        dists = np.zeros((npts, 3))
        dists[:, 0] = X.ravel()
        dists[:, 1] = Y.ravel()

        sigmas = np.ones((npts, 3, 3))
        sigmas[:, 0, 0] = uxLengthScale
        sigmas[:, 0, 1] = uyLengthScale
        sigmas[:, 1, 0] = vxLengthScale
        sigmas[:, 1, 1] = vyLengthScale
        sigmas[:, 2, 0] = wxLengthScale
        sigmas[:, 2, 1] = wyLengthScale

        levels = np.linspace(-1, 1, 31)

        sigs = blob(dists, sigmas)
        fig, ax = plt.subplots()
        triang = matplotlib.tri.Triangulation(X.ravel(), Y.ravel())
        zero = np.less(np.abs(sigs[:, 0]), 1e-16)
        mask = np.all(np.where(zero[triang.triangles], True, False), axis=1)
        triang.set_mask(mask)
        tcf = ax.tricontourf(triang, sigs[:, 0], levels=levels)
        ax.scatter([0], [0], color="k", zorder=4)
        cb = fig.colorbar(tcf, ticks=np.linspace(-1, 1, 11))
        cb.set_label(r"$u^{*}$" + " Contribution")
        ax.set_title(
            r"$\sigma_{ux}=$"
            + f"{uxLengthScale}, "
            + r"$\sigma_{uy}=$"
            + f"{uyLengthScale}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(linestyle="--")
        plt.show()

        fig, ax = plt.subplots()
        mask = np.all(np.where(zero[triang.triangles], True, False), axis=1)
        triang.set_mask(mask)
        tcf = ax.tricontourf(triang, sigs[:, 1], levels=levels)
        ax.scatter([0], [0], color="k", zorder=4)
        cb = fig.colorbar(tcf, ticks=np.linspace(-1, 1, 11))
        cb.set_label(r"$v^{*}$" + " Contribution")
        ax.set_title(
            r"$\sigma_{vx}$="
            + f"{vxLengthScale}, "
            + r"$\sigma_{vy}=$"
            + f"{vyLengthScale}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(linestyle="--")
        plt.show()

        fig, ax = plt.subplots()
        mask = np.all(np.where(zero[triang.triangles], True, False), axis=1)
        triang.set_mask(mask)
        tcf = ax.tricontourf(triang, sigs[:, 2], levels=levels)
        ax.scatter([0], [0], color="k", zorder=4)
        cb = fig.colorbar(tcf, ticks=np.linspace(-1, 1, 11))
        cb.set_label(r"$w^{*}$" + " Contribution")
        ax.set_title(
            r"$\sigma_{wx}$="
            + f"{wxLengthScale}, "
            + r"$\sigma_{wy}=$"
            + f"{wyLengthScale}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(linestyle="--")
        plt.show()
