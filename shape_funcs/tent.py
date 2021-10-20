import numpy as np


def tent(dists, sigmas):
    """

    The legendary tent function. NOTE: distances in dists are not absolute valued.
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

    shape = (dists.shape[0], 1)

    # Individual f(x) 'tent' functions for each contributing point
    fxx = np.sqrt(1.5) * (
        1.0 - np.abs(np.reshape(dists[:, 0], shape)) / sigmas[:, :, 0]
    )
    fxy = np.sqrt(1.5) * (
        1.0 - np.abs(np.reshape(dists[:, 1], shape)) / sigmas[:, :, 1]
    )
    fxz = np.sqrt(1.5) * (
        1.0 - np.abs(np.reshape(dists[:, 2], shape)) / sigmas[:, :, 2]
    )

    np.clip(fxx, 0.0, None, out=fxx)
    np.clip(fxy, 0.0, None, out=fxy)
    np.clip(fxz, 0.0, None, out=fxz)

    # Total f(x) from each contributing point
    fx = fxx * fxy * fxz

    # For tracking purposes, see if this point has zero contributions
    tent.empty = np.any((np.sum(fx, axis=0) == 0.0))

    return fx


if __name__ == "__main__":

    nsig = 201
    ux_length_scale = 1.0
    uy_length_scale = 0.75
    vx_length_scale = 0.5
    vy_length_scale = 0.25
    wx_length_scale = 0.25
    wy_length_scale = 0.25

    line = np.linspace(-ux_length_scale * 1.5, ux_length_scale * 1.5, nsig)

    dists = np.zeros((nsig, 3))
    dists[:, 0] = line

    sigmas = np.ones((nsig, 3, 3))
    sigmas[:, 0, 0] = ux_length_scale
    sigmas[:, 1, 0] = vx_length_scale
    sigmas[:, 2, 0] = wx_length_scale

    sigs = tent(dists, sigmas)

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
    plt.rcParams["image.cmap"] = "Blues"

    fig, ax = plt.subplots()
    lo = ax.plot(line, sigs)
    ax.set_title(r"$f(|x-\sigma_{i}|)$")
    ax.set_xlabel(r"$x$")
    plt.legend(
        lo,
        (
            r"$\sigma_{u}$=" + f"{ux_length_scale}",
            r"$\sigma_{v}=$" + f"{vx_length_scale}",
            r"$\sigma_{w}=$" + f"{wx_length_scale}",
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
        sigmas[:, 0, 0] = ux_length_scale
        sigmas[:, 0, 1] = uy_length_scale
        sigmas[:, 1, 0] = vx_length_scale
        sigmas[:, 1, 1] = vy_length_scale
        sigmas[:, 2, 0] = wx_length_scale
        sigmas[:, 2, 1] = wy_length_scale

        levels = np.linspace(0, 2, 31)
        ticks = np.linspace(0, 2, 11)

        sigs = tent(dists, sigmas)
        fig, ax = plt.subplots()
        triang = matplotlib.tri.Triangulation(X.ravel(), Y.ravel())
        zero = np.less(np.abs(sigs[:, 0]), 1e-16)
        mask = np.all(np.where(zero[triang.triangles], True, False), axis=1)
        triang.set_mask(mask)
        tcf = ax.tricontourf(triang, sigs[:, 0], levels=levels)
        ax.scatter([0], [0], color="k", zorder=4)
        cb = fig.colorbar(tcf, ticks=ticks)
        cb.set_label(r"$u^{*}$" + " Contribution")
        ax.set_title(
            r"$\sigma_{ux}=$"
            + f"{ux_length_scale}, "
            + r"$\sigma_{uy}=$"
            + f"{uy_length_scale}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(linestyle="--")
        plt.show()

        fig, ax = plt.subplots()
        zero = np.less(np.abs(sigs[:, 1]), 1e-16)
        mask = np.all(np.where(zero[triang.triangles], True, False), axis=1)
        triang.set_mask(mask)
        tcf = ax.tricontourf(triang, sigs[:, 1], levels=levels)
        ax.scatter([0], [0], color="k", zorder=4)
        cb = fig.colorbar(tcf, ticks=ticks)
        cb.set_label(r"$v^{*}$" + " Contribution")
        ax.set_title(
            r"$\sigma_{vx}$="
            + f"{vx_length_scale}, "
            + r"$\sigma_{vy}=$"
            + f"{vy_length_scale}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(linestyle="--")
        plt.show()

        fig, ax = plt.subplots()
        zero = np.less(np.abs(sigs[:, 2]), 1e-16)
        mask = np.all(np.where(zero[triang.triangles], True, False), axis=1)
        triang.set_mask(mask)
        tcf = ax.tricontourf(triang, sigs[:, 2], levels=levels)
        ax.scatter([0], [0], color="k", zorder=4)
        cb = fig.colorbar(tcf, ticks=ticks)
        cb.set_label(r"$w^{*}$" + " Contribution")
        ax.set_title(
            r"$\sigma_{wx}$="
            + f"{wx_length_scale}, "
            + r"$\sigma_{wy}=$"
            + f"{wy_length_scale}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(linestyle="--")
        plt.show()
