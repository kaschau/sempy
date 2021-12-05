import sys
import numpy as np
from scipy import interpolate as itrp
from .misc import progressBar
from . import normalization as norm
from . import shapeFuncs


def generatePrimes(
    ys,
    zs,
    domain,
    nframes,
    normalization,
    shape="tent",
    interpolate=False,
    convect="uniform",
    progress=True,
):
    """
    Generate Primes Function

    Routine to march down the length (time-line) of mega-box at specified (y,z) coordinate points, stopping along the way
    to calculate the fluctuations at each time point. To improve performace, we first filter out all the eddy in domain
    that cannot possibly contribute to the group of (y,z) points provided based on the eddy's sigmas in y and z.

    Further, for each (y,z) pair we want to compute a signal for, we filter out the eddys that cannot possible contribute
    to the current (y,z) point's time line, again based on the eddy's sigmas in y and z.

    Finally, we can filter out the eddys that do not contribute to an individual point as we march down the time line
    based on the reduced set of eddy's sigmas in x.

    With this very reduced set of eddys that contribute to a point, we compute the fluctuation for that point.


    Parameters:
    -----------
      ys     : numpy.array
            Array of y coordinates of any shape
      zs     : numpy.array
            Array of shape(y) corresponsing one-to-one matching (y,z) pairs
      domain : sempy.domain
            Domain object fully populated with parameters and data.
      nframes : int
            Number of frames to generate per (y,z) pair
      normalization : str
            String corresponding to the method of normalizing the signal.
            Available options are:
               'jarrin' : Approximates an integral with Vbox/(neddy*V_eddy)
               'exact'  : Produce exact statistics by bending the signal to your will
               'none'   : Return the raw sum of the eddys for each point
      interpolation : bool
            T/F flag to determine whether your frame rate is high enough to approximate the
            continuous signal, thus no significalt loss of statistics, or if you plan to interpolate
            between frames. If True, a time signal is "pre-interpolated" and statistics are imposed
            on that pre-interpolated signal, then just the frame values are returned. If False, nothing
            is done
      convect : str
            String corresponding to the method of convection through the flow.
            Current options are:
               'uniform' : Convect through mega-volume at Ublk everywhere
               'local'   : Convect through mega-volume at local convective speed based on domain.Ubar_interp(y)
      progress : bool
            Whether to display progress bar

    Returns:
    --------
        up,vp,wp : numpy.arrrays
            Fluctuation result arrays of shape ( (nframes, shape(y)) ). First axis is 'time', second axis is the
            shape of the input ys and zs arrays.
            So up[0] is the corresponding u fluctiations at all the y,z points for the first frame.
    """

    # check if we have eddys or not
    if domain.eddyLocs is None:
        raise RuntimeError(
            "Please populate your domain before trying to generate fluctiations"
        )

    # Check that nframes is large enough with exact normalization
    if 3 < nframes < 10 and normalization == "exact":
        print(
            "WARNING: You are using exact normalization with very few framses. Weird things may happen. Consider using jarrin normalization or increasing number of framses."
        )
    elif nframes <= 3 and normalization == "exact":
        raise RuntimeError(
            "Need more frames to use exact normaliation, consider using jarrin or creating more frames."
        )

    # Check ys and zs are the same shape
    if ys.shape != zs.shape:
        raise RuntimeError("ys and zs need to be the same shape.")

    # make sure ys and zs are numpy arrays
    ys = np.array(ys)
    zs = np.array(zs)

    # As a sanity check, make sure none of our points are outside the patch
    if (
        ys.min() < -0.0001 * domain.yHeight
        or ys.max() > 1.0001 * domain.yHeight
        or zs.min() < -0.0001 * domain.zWidth
        or zs.max() > 1.0001 * domain.zWidth
    ):
        raise ValueError(
            "Woah there, some of your points you are trying to calculate fluctuations for are completely outside your domain!"
        )

    # store the input array shape and then flatten the yz pairs
    yshape = ys.shape
    ys = ys.ravel()
    zs = zs.ravel()

    # We want to filter out all eddys that are not possibly going to contribute to this set of ys and zs
    # we are going to refer to the bounding box that encapsulates ALL y,z pairs of the current 'patch'
    patchYmin = ys.min()
    patchYmax = ys.max()
    patchZmin = zs.min()
    patchZmax = zs.max()

    eddysInPatch = np.where(
        (domain.eddyLocs[:, 1] - np.max(domain.sigmas[:, :, 1], axis=1) < patchYmax)
        & (domain.eddyLocs[:, 1] + np.max(domain.sigmas[:, :, 1], axis=1) > patchYmin)
        & (domain.eddyLocs[:, 2] - np.max(domain.sigmas[:, :, 2], axis=1) < patchZmax)
        & (domain.eddyLocs[:, 2] + np.max(domain.sigmas[:, :, 2], axis=1) > patchZmin)
    )
    eddyLocsInPatch = domain.eddyLocs[eddysInPatch]
    sigmasInPatch = domain.sigmas[eddysInPatch]
    epsInPatch = domain.eps[eddysInPatch]

    ######################################################################
    # We now have a reduced set of eddys that overlap the current patch
    ######################################################################

    # Storage for fluctuations
    up = np.empty((nframes, len(ys)))
    vp = np.empty((nframes, len(ys)))
    wp = np.empty((nframes, len(ys)))

    # just counter for progress display
    total = len(ys)

    # Define "time" points for frames, if we are using uniform
    # convection, we only need to do this once.
    if convect == "uniform":
        xs = np.linspace(0, domain.xLength, nframes)

    # Loop over each location in current patch
    for i, (y, z) in enumerate(zip(ys, zs)):

        zeroOnline = False
        if progress:
            progressBar(i + 1, total, "Generating Primes")

        # Find eddies that contribute on the current y,z line. This search is done on the reduced set of
        # eddys filtered on the "patch"
        eddysOnLine = np.where(
            (np.abs(eddyLocsInPatch[:, 1] - y) < np.max(sigmasInPatch[:, :, 1], axis=1))
            & (
                np.abs(eddyLocsInPatch[:, 2] - z)
                < np.max(sigmasInPatch[:, :, 2], axis=1)
            )
        )
        eddyLocsOnLine = eddyLocsInPatch[eddysOnLine]
        sigmasOnLine = sigmasInPatch[eddysOnLine]
        epsOnLine = epsInPatch[eddysOnLine]

        # We want to know if an entire line has zero eddys, this will be annoying for BL in the free stream
        # so we will only print out a warning for the BL cases if the value of y is below the BL thickness
        if len(eddyLocsOnLine) == 0:
            zeroOnline = True
            if domain.flowType != "bl" or y < domain.delta:
                print(f"Warning, no eddys detected on entire time line at y={y},z={z}")

        ######################################################################
        # We now have a reduced set of eddys that overlap the current y,z line
        ######################################################################

        # Define "time" points for frames, if we are using
        # local convection we need to recalculate for each y location.
        if convect == "local":
            localUbar = domain.ubarInterp(y)
            length = domain.xLength * localUbar / domain.Ublk
            xs = np.linspace(0, length, nframes)

        # Storage for un-normalized fluctuations
        primesNoNorm = np.zeros((xs.shape[0], 3))

        emptyPts = 0  # counter for empty points

        # If this line has no eddys on it, move on
        if zeroOnline:
            continue

        if convect == "local":
            # We need each eddys individual Ubar for the offset calculated below
            localEddyUbar = domain.ubarInterp(eddyLocsOnLine[:, 1])

        # Travel down line at this y,z location
        for j, x in enumerate(xs):
            zeroOnpoint = False

            if convect == "local":
                # This may be tough to explain, but just draw it out for yourself and you'll
                # figure it out:

                # If we want each eddy to convect with its own local velocity
                # instead of the Ublk velocity, it is not enough to just traverse through the
                # mega box at the profile Ubar for the current location's y height. This is
                # because the eddys located slightly above/below the location we are
                # traversing down (that contribute to fluctuations) are
                # moving at different speeds than our current point of interest. So we need to
                # calculate an offset to account for that fact that as we have traversed through the
                # domain at the local convective speed, the faster eddys will approach
                # us slightly more quickly, while the slower eddys
                # will approach us more slowly. This x offset will account for that and is merely the difference
                # in convection speeds between the current line we are traversing down, and the speeds of
                # the individual eddys.
                xOffset = (localUbar - localEddyUbar) / localUbar * x
            else:
                # If all the eddys convect at the same speed, then traversing through the mega box at Ublk
                # is identical so the eddys convecting by us at Ublk, so there is no need to offset any
                # eddy positions
                xOffset = np.zeros(eddyLocsOnLine.shape[0])

            #########################
            # Compute the fluctuations
            #########################
            # Find all non zero eddies for at current time "x"
            xDist = np.abs((eddyLocsOnLine[:, 0] + xOffset) - x)
            eddysOnPoint = np.where(xDist < np.max(sigmasOnLine[:, :, 0], axis=1))
            xOffset = xOffset[eddysOnPoint]
            if len(eddysOnPoint[0]) == 0:
                emptyPts += 1
                continue

            ###################################################################
            # We now have a reduced set of eddys that overlap the current point
            ###################################################################

            # Compute distances to all contributing eddys
            dists = np.empty((len(eddysOnPoint[0]), 3))
            dists[:, 0] = eddyLocsOnLine[eddysOnPoint][:, 0] + xOffset - x
            dists[:, 1] = eddyLocsOnLine[eddysOnPoint][:, 1] - y
            dists[:, 2] = eddyLocsOnLine[eddysOnPoint][:, 2] - z

            # Collect sigmas from all contributing points
            sigmasOnPoint = sigmasOnLine[eddysOnPoint]

            # Compute the fluctuation contributions of each eddy, for each component
            # via a "shape function"
            if shape == "tent":
                fx = shapeFuncs.tent(dists, sigmasOnPoint)
                if shapeFuncs.tent.empty:
                    emptyPts += 1
            elif shape == "blob":
                fx = shapeFuncs.blob(dists, sigmasOnPoint)
            else:
                raise NameError(f"Error: Unknown shape function : {shape}")

            # We have to do this here for jarrin even though its ugly AF
            if normalization == "jarrin":
                fx = 1.0 / np.sqrt(np.product(sigmasOnPoint, axis=2)) * fx

            # multiply each eddys function/component by its sign
            primesNoNorm[j, :] = np.sum(epsOnLine[eddysOnPoint] * fx, axis=0)

        # We will warn the user if we detect more that 10 empty points along this line.a
        if emptyPts > 10:
            print(
                f"Warning, {emptyPts} points with zero fluctuations detected at y={y},z={z}\n"
            )

        ################################################################
        # We now have un-normalized fluctuation data over the entire time
        # series at this (y.z) locaiton
        ################################################################

        # If we are planning on interpolating the signal for a raptor simulaiton, we need to
        # approximate the interpolation here, then normalize the interpolated signal.
        # Otherwise the stats of the interpolated signal in raptor will under represent the
        # desired statistics.
        if interpolate:
            # Current we approximate with 10 points between frames. Could be experimented with
            ptsBtwFrames = 10
            tempN = [j for j in range(nframes)]
            primesInterp = itrp.CubicSpline(
                tempN, primesNoNorm, bc_type="not-a-knot", axis=0
            )
            intrpN = np.linspace(0, nframes - 1, (nframes - 1) * (ptsBtwFrames + 1) + 1)
            primesNoNorm = primesInterp(intrpN)
            frameIndicies = tuple([j * (ptsBtwFrames + 1) for j in range(nframes)])
        else:
            frameIndicies = tuple([j for j in range(nframes)])

        # Normalize the time signals
        if normalization == "exact":
            primesNormed = norm.exactNorm(primesNoNorm)
        elif normalization == "jarrin":
            primesNormed = norm.jarrinNorm(primesNoNorm, domain)
        elif normalization == "none":
            primesNormed = primesNoNorm
        else:
            raise NameError(f"Error: Unknown normalization : {normalization}")

        # Compute Rij for current y location
        Rij = domain.rijInterp(y)
        # Cholesky decomp of stats
        L = np.linalg.cholesky(Rij)
        # Multiply normalized signal by stats
        prime = np.matmul(L, primesNormed.T).T

        # Keep only the points on the frames
        up[:, i] = prime[frameIndicies, 0]
        vp[:, i] = prime[frameIndicies, 1]
        wp[:, i] = prime[frameIndicies, 2]

    # Return fluctuations
    up = up.reshape(tuple([nframes] + list(yshape)))
    vp = vp.reshape(tuple([nframes] + list(yshape)))
    wp = wp.reshape(tuple([nframes] + list(yshape)))

    return up, vp, wp
