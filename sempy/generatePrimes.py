import numpy as np
from scipy import interpolate as itrp

from . import normalization as norm
from . import shapeFuncs
from .misc import progressBar


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

    Routine to march down the length (time-line) of mega-box at specified (y,z)
    coordinate points, stopping along the way to calculate the fluctuations at
    each time point. To improve performace, we first filter out all the eddy in
    domain that cannot possibly contribute to the group of (y,z) points
    provided based on the eddy's sigmas in y and z.

    Further, for each (y,z) pair we want to compute a signal for, we filter
    out the eddys that cannot possible contribute to the current (y,z) point's
    time line, again based on the eddy's sigmas in y and z.

    Finally, we can filter out the eddys that do not contribute to an
    individual point as we march down the time line based on the reduced set of
    eddy's sigmas in x.

    With this very reduced set of eddys that contribute to a point, we compute
    the fluctuation for that point.


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
               'exact'  : Produce exact statistics by bending the signal to
                          your will
               'none'   : Return the raw sum of the eddys for each point
      interpolation : bool
            T/F flag to determine whether your frame rate is high enough to
            approximate the continuous signal, thus no significalt loss of
            statistics, or if you plan to interpolate between frames. If True,
            a time signal is "pre-interpolated" and statistics are imposed
            on that pre-interpolated signal, then just the frame values are
            returned. If False, nothing is done
      convect : str
            String corresponding to the method of convection through the flow.
            Current options are:
               'uniform' : Convect through mega-volume at Uo everywhere
               'local'   : Convect through mega-volume at local convective
                           speed based on domain.Ubar_interp(y)
      progress : bool
            Whether to display progress bar

    Returns:
    --------
        up,vp,wp : numpy.arrrays
            Fluctuation result arrays of shape ( (nframes, shape(y)) ).
            First axis is 'time', second axis is the shape of the input ys and
            zs arrays. So up[0] is the corresponding u fluctiations at all the
            y,z points for the first frame.
    """

    # check if we have eddys or not
    if domain.eddyLocs is None:
        raise RuntimeError(
            "Please populate your domain before trying to generate fluctuations"
        )

    # Validate keyword options up front
    if convect not in ("uniform", "local"):
        raise NameError(f"Error: Unknown convection method : {convect}")
    if shape not in ("tent", "blob"):
        raise NameError(f"Error: Unknown shape function : {shape}")
    if normalization not in ("exact", "jarrin", "none"):
        raise NameError(f"Error: Unknown normalization : {normalization}")

    # Check that nframes is large enough with exact normalization
    if 3 < nframes < 10 and normalization == "exact":
        print(
            "WARNING: You are using exact normalization with very few framses.\n",
            "Weird things may happen. Consider using jarrin normalization\n",
            "or increasing number of framses.\n",
        )
    elif nframes <= 3 and normalization == "exact":
        raise RuntimeError(
            "Need more frames to use exact normaliation,\n",
            "consider using jarrin or creating more frames.\n",
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
            "Woah there, some of your points you are trying to calculate\n",
            "fluctuations for are completely outside your domain!",
        )

    # store the input array shape and then flatten the yz pairs
    yshape = ys.shape
    ys = ys.ravel()
    zs = zs.ravel()

    # We want to filter out all eddys that are not possibly going to
    # contribute to this set of ys and zs we are going to refer to
    # the bounding box that encapsulates ALL y,z pairs of the current 'patch'
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

    # Each eddy's largest y and z sigma is reused for every (y,z) point,
    # so compute them once here rather than inside the loop below
    maxSigmaXInPatch = np.max(sigmasInPatch[:, :, 0], axis=1)
    maxSigmaYInPatch = np.max(sigmasInPatch[:, :, 1], axis=1)
    maxSigmaZInPatch = np.max(sigmasInPatch[:, :, 2], axis=1)

    # Sort the patch's eddys by y location so that each (y,z) point below
    # only needs to examine the window of eddys that can possibly reach it
    # (found by binary search) instead of every eddy in the patch
    srt = np.argsort(eddyLocsInPatch[:, 1])
    eddyLocsInPatch = eddyLocsInPatch[srt]
    sigmasInPatch = sigmasInPatch[srt]
    epsInPatch = epsInPatch[srt]
    maxSigmaXInPatch = maxSigmaXInPatch[srt]
    maxSigmaYInPatch = maxSigmaYInPatch[srt]
    maxSigmaZInPatch = maxSigmaZInPatch[srt]
    eddyYsInPatch = eddyLocsInPatch[:, 1]
    # No eddy in the patch reaches further in y than this
    if eddyYsInPatch.shape[0] > 0:
        sigmaYBound = np.max(maxSigmaYInPatch)
    else:
        sigmaYBound = 0.0

    # Which frames an eddy contributes to depends only on the eddy itself,
    # not on the (y,z) point: frame j occurs at time t_j = j*dt no matter
    # the convection method, and eddy e crosses a probe during
    #
    #     t in ( (x_e - maxSigmaX_e)/ubar_e , (x_e + maxSigmaX_e)/ubar_e )
    #
    # (with ubar_e = Uo everywhere for uniform convection). Since frame
    # times are sorted, each eddy touches a contiguous run of frames found
    # once here with a binary search, rather than testing every eddy
    # against every frame for every point.
    if convect == "local":
        ubarInPatch = domain.ubarInterp(eddyYsInPatch)
    else:
        ubarInPatch = np.full(eddyYsInPatch.shape[0], domain.Uo)
    frameTimes = np.linspace(0, domain.xLength / domain.Uo, nframes)
    with np.errstate(divide="ignore", invalid="ignore"):
        tLo = (eddyLocsInPatch[:, 0] - maxSigmaXInPatch) / ubarInPatch
        tHi = (eddyLocsInPatch[:, 0] + maxSigmaXInPatch) / ubarInPatch
    # An eddy with zero ubar (at the wall, where the profile clamps to
    # zero) never convects past us, so it is considered at every frame;
    # its shape function still zeroes out any frame it cannot reach
    degenerate = ubarInPatch <= 0.0
    if np.any(degenerate):
        tLo[degenerate] = -np.inf
        tHi[degenerate] = np.inf
    jLoInPatch = np.searchsorted(frameTimes, tLo, "right")
    jHiInPatch = np.searchsorted(frameTimes, tHi, "left")

    ######################################################################
    # We now have a reduced set of eddys that overlap the current patch
    ######################################################################

    # Storage for fluctuations (zeros, not empty, so that lines with no
    # eddys on them return zero fluctuations instead of garbage memory)
    up = np.zeros((nframes, len(ys)))
    vp = np.zeros((nframes, len(ys)))
    wp = np.zeros((nframes, len(ys)))

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

        # Find eddies that contribute on the current y,z line. This search
        # is done on a conservative y window of the eddys filtered on the
        # "patch" (the exact test is unchanged)
        yLo = np.searchsorted(eddyYsInPatch, y - sigmaYBound, side="left")
        yHi = np.searchsorted(eddyYsInPatch, y + sigmaYBound, side="right")
        eddysOnLine = np.where(
            (np.abs(eddyYsInPatch[yLo:yHi] - y) < maxSigmaYInPatch[yLo:yHi])
            & (np.abs(eddyLocsInPatch[yLo:yHi, 2] - z) < maxSigmaZInPatch[yLo:yHi])
        )
        # Absolute patch indices of the eddys on this line; the heavy
        # per-pair data (sigmas, eps) is gathered straight from the patch
        # arrays further below instead of being copied out per line
        lineIdx = yLo + eddysOnLine[0]
        eddyXsOnLine = eddyLocsInPatch[lineIdx, 0]
        eddyYsOnLine = eddyLocsInPatch[lineIdx, 1]
        eddyZsOnLine = eddyLocsInPatch[lineIdx, 2]

        # We want to know if an entire line has zero eddys, this will
        # be annoying for BL in the free stream so we will only print
        # out a warning for the BL cases if the value of y is below the
        # BL thickness
        if len(lineIdx) == 0:
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
            length = domain.xLength * localUbar / domain.Uo
            xs = np.linspace(0, length, nframes)

        # Storage for un-normalized fluctuations
        primesNoNorm = np.zeros((xs.shape[0], 3))

        emptyPts = 0  # counter for empty points

        # If this line has no eddys on it, move on
        if zeroOnline:
            continue

        if convect == "local":
            # We need each eddys individual Ubar for the offset
            # calculated below.

            # This may be tough to explain, but just draw it out for
            # yourself and you'll figure it out:

            # If we want each eddy to convect with its own local velocity
            # instead of the Uo velocity, it is not enough to just traverse
            # through the mega box at the profile Ubar for the current
            # location's y height. This is because the eddys located
            # slightly above/below the location we are
            # traversing down (that contribute to fluctuations) are
            # moving at different speeds than our current point of
            # interest. So we need to calculate an offset to account for
            # that fact that as we have traversed through the
            # domain at the local convective speed, the faster eddys will
            # approach us slightly more quickly, while the slower eddys
            # will approach us more slowly. This x offset accounts for
            # that and is merely the difference in convection speeds
            # between the current line we are traversing down, and the
            # speeds of the individual eddys.
            localEddyUbar = ubarInPatch[lineIdx]

        # Rather than travel down the line one frame at a time testing
        # every eddy, look up each eddy's precomputed contiguous run of
        # contributing frames, then evaluate all (eddy, frame)
        # contribution pairs at once and accumulate them into their
        # frames with bincount.
        jLo = jLoInPatch[lineIdx]
        jHi = jHiInPatch[lineIdx]
        pairCounts = jHi - jLo
        cumCounts = np.cumsum(pairCounts)
        totalPairs = int(cumCounts[-1])

        # Track the total shape function contribution per frame to detect
        # empty points below
        fxTotal = np.zeros((xs.shape[0], 3))

        # Process the (eddy, frame) pairs in chunks to cap peak memory
        maxPairsPerChunk = 5_000_000
        nOnLine = eddyXsOnLine.shape[0]
        avgPairs = max(1.0, totalPairs / nOnLine)
        eddyStep = max(1, int(maxPairsPerChunk / avgPairs))
        for s in range(0, nOnLine, eddyStep):
            e = min(s + eddyStep, nOnLine)
            counts = pairCounts[s:e]
            chunkPairs = int(np.sum(counts))
            if chunkPairs == 0:
                continue

            # Expand each eddy's contiguous run of frame indices
            eddyIdx = np.repeat(np.arange(s, e), counts)
            runStarts = np.cumsum(counts) - counts
            frameIdx = np.repeat(jLo[s:e] - runStarts, counts) + np.arange(chunkPairs)
            x = xs[frameIdx]

            if convect == "local":
                xOffset = (localUbar - localEddyUbar[eddyIdx]) / localUbar * x
            else:
                xOffset = np.zeros(chunkPairs)

            # Compute distances of every contributing (eddy, frame) pair
            dists = np.empty((chunkPairs, 3))
            dists[:, 0] = eddyXsOnLine[eddyIdx] + xOffset - x
            dists[:, 1] = eddyYsOnLine[eddyIdx] - y
            dists[:, 2] = eddyZsOnLine[eddyIdx] - z

            # Collect sigmas of all contributing pairs
            pairIdx = lineIdx[eddyIdx]
            sigmasOnPoint = sigmasInPatch[pairIdx]

            # Compute the fluctuation contributions of each eddy, for each
            # component via a "shape function"
            if shape == "tent":
                fx = shapeFuncs.tent(dists, sigmasOnPoint)
            elif shape == "blob":
                fx = shapeFuncs.blob(dists, sigmasOnPoint)

            # We have to do this here for jarrin even though its ugly AF
            if normalization == "jarrin":
                contrib = 1.0 / np.sqrt(np.prod(sigmasOnPoint, axis=2)) * fx
            else:
                contrib = fx

            # multiply each eddys function/component by its sign and
            # accumulate everything into its frame
            contrib = epsInPatch[pairIdx] * contrib
            for comp in range(3):
                primesNoNorm[:, comp] += np.bincount(
                    frameIdx, weights=contrib[:, comp], minlength=xs.shape[0]
                )
                fxTotal[:, comp] += np.bincount(
                    frameIdx, weights=fx[:, comp], minlength=xs.shape[0]
                )

        # Frames with no eddys, or where the shape functions summed to
        # zero contribution in some component, count as empty points
        emptyPts = int(np.sum(np.any(fxTotal == 0.0, axis=1)))

        # We will warn the user if we detect more than
        # 10 empty points along this line.
        if emptyPts > 10:
            print(
                f"Warning, {emptyPts} points with zero fluctuations detected at y={y},z={z}\n"
            )

        ################################################################
        # We now have un-normalized fluctuation data over the entire time
        # series at this (y.z) locaiton
        ################################################################

        # If we are planning on interpolating the signal for a simulaiton,
        # we need to approximate the interpolation here, then normalize the
        # interpolated signal. Otherwise the stats of the interpolated signal
        # in situ will under represent the desired statistics.
        if interpolate:
            # Current we approximate with 10 points between frames.
            # Could be experimented with
            ptsBtwFrames = 10
            primesInterp = itrp.CubicSpline(
                np.arange(nframes), primesNoNorm, bc_type="not-a-knot", axis=0
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
