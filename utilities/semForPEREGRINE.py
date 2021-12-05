#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility generates files full of cubic spline coefficients representing a turbulent signal
that can be applied at the inlet of a RAPTOR simulation. Just call this utility with
an input file. There must also be a file where this utility is run called : patches.txt
This patches.txt maps the patch number to a RAPTOR block number. The format of the file is
   Patch #, Block #
Where each line represents a patch<=>block pair.

Example
-------
/path/to/sempy/utilities/sem4raptor.py <sem.inp>

The output is a directory with the name of your input file, +'_alphas' full of your alpha coeffs.

You can also run this utility in parallel, just use mpiexec

Example
-------
mpiexec -np <np> /path/to/sempy/utilities/sem4raptor.py <sem.inp>

"""

from mpi4py import MPI
import sys
import os
import peregrinepy as pg
import sempy
import numpy as np
import yaml
from scipy import interpolate as itrp

inputFile = sys.argv[1]

# Initialize the parallel information for each rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Use the input file name as the output directory
outputDir = inputFile.split(".")[0] + "_alphas"
if rank == 0:
    if os.path.exists(outputDir):
        pass
    else:
        os.makedirs(outputDir)
############################################################################################
# Read in SEM parameters
############################################################################################
# The zeroth rank will read in the input file and give it to the other ranks
if rank == 0:
    with open(inputFile, "r") as f:
        seminp = yaml.load(f, Loader=yaml.FullLoader)
else:
    seminp = None
seminp = comm.bcast(seminp, root=0)

if seminp["nframes"] > 9999:
    raise ValueError(
        "Error: Currently, the interpolation procedure in RAPTOR is only setup for nframes < 9999"
    )


############################################################################################
# Create the domain based on above inputs
############################################################################################
# Initialize domain
domain = sempy.geometries.box(
    seminp["domainType"],
    seminp["Ublk"],
    seminp["totalTime"],
    seminp["yHeight"],
    seminp["zWidth"],
    seminp["delta"],
    seminp["utau"],
    seminp["viscosity"],
)

# Set flow properties from existing data
domain.setSemData(
    sigmasFrom=seminp["sigmasFrom"],
    statsFrom=seminp["statsFrom"],
    profileFrom=seminp["profileFrom"],
    scaleFactor=seminp["scaleFactor"],
)

# Only the zeroth rank populates the domain
if rank == 0:
    # Populate the domain
    domain.populate(seminp["cEddy"], seminp["populationMethod"])
    # Create the eps
    domain.generateEps()
    # Compute sigmas
    domain.computeSigmas()
    # Make it periodic
    domain.makePeriodic(
        periodicX=seminp["periodicX"],
        periodicY=seminp["periodicY"],
        periodicZ=seminp["periodicZ"],
    )
    tempNeddy = domain.neddy
else:
    tempNeddy = None
# We have to overwrite the worker processes' neddy property
# so that each worker know how many eddys are in the actual
# domain
tempNeddy = comm.bcast(tempNeddy, root=0)
if rank != 0:
    domain._neddy = tempNeddy

############################################################################################
# Read in patches.txt
############################################################################################
# Only the zeroth rank reads in patches.txt
if rank == 0:
    patchNum = []
    blockNum = []
    with open("patches.txt", "r") as f:
        for line in [i for i in f.readlines() if not i.startswith("#")]:
            li = line.strip().split(",")
            patchNum.append(int(li[0]))
            blockNum.append(int(li[1]))

    npatches = len(patchNum)
    # Now we assign patches to ranks
    if size > npatches:
        print(
            "\n\nFYI, patches are assigned to processors, so using more processors than patches gives you no performace increase.\n\n"
        )
        maxRanksPerBlock = 1
    else:
        maxRanksPerBlock = int(npatches / size) + (1 if npatches % size > 0 else 0)
    allRankPatches = [[None for j in range(maxRanksPerBlock)] for i in range(size)]
    allRankBlocks = [[None for j in range(maxRanksPerBlock)] for i in range(size)]
    i = 0
    j = 0
    for bn, pn in zip(blockNum, patchNum):
        allRankPatches[i][j] = pn
        allRankBlocks[i][j] = bn
        i += 1
        if i == size:
            i = 0
            j += 1
    myPatchNums = allRankPatches[0]
    myBlockNums = allRankBlocks[0]
    # Send the list of patch numbers to the respective ranks ranks
    for i, rsp in enumerate(allRankPatches[1::]):
        comm.send(rsp, dest=i + 1, tag=11)
else:
    myPatchNums = comm.recv(source=0, tag=11)


############################################################################################
# PEREGRINE stuff (read in grid and make the patches)
############################################################################################
# Only rank ones reads in the grid
if rank == 0:
    nblks = len(
        [
            f
            for f in os.listdir(seminp["gridPath"])
            if f.startswith("gv.") and f.endswith(".h5")
        ]
    )
    if nblks == 0:
        raise ValueError(f'Cant find any grid files in {seminp["gridPath"]}')

    mb = pg.multiBlock.grid(nblks)
    pg.readers.read_raptor_grid(mb, seminp["gridPath"])
    # Flip the grid if it is oriented upside down in the true grid
    if seminp["flipdom"]:
        for bn in blockNum:
            blk = mb[bn - 1]
            blk.y = -blk.y
            blk.z = -blk.z
    # Determine the extents the grid needs to be shifted
    ymin = np.inf
    zmin = np.inf
    for bn in blockNum:
        blk = mb[bn - 1]
        ymin = min(ymin, blk.array["y"][0, :, :].min())
        zmin = min(zmin, blk.array["z"][0, :, :].min())
    # Now shift the grid to match the domain (0,0) starting point
    for bn in blockNum:
        blk = mb[bn - 1]
        blk.y = blk.array["y"] - ymin
        blk.z = blk.array["z"] - zmin
        # Compute the locations of face centers
        blk.computeMetrics(fdOrder=2)

# Progress bar gets messsy with all the blocks, so we only show progress with rank 0
if rank == 0:
    progress = True
else:
    progress = False

if progress:
    # Print out a summarry
    print(domain)
    print(
        f'Generating signal that is {seminp["totalTime"]} [s] long, with {seminp["nframes"]} frames using {seminp["convect"]} convection speed.\n'
    )
    print(
        "\n*******************************************************************************"
    )
    print(
        "**************************** Generating Primes ********************************"
    )
    print(
        "*******************************************************************************\n"
    )

for i, pn in enumerate(myPatchNums):
    if rank != 0 and pn is None:
        continue
    # If we are the zeroth block, then we compute and send the patches to all the other blocks
    if rank == 0:
        for sendRank in [j + 1 for j in range(size - 1)]:
            sendPatchNum = allRankPatches[sendRank][i]
            sendBlockNum = allRankBlocks[sendRank][i]
            if sendPatchNum is None:
                continue
            blk = mb[sendBlockNum - 1]
            # Create the patch for this block
            yc = blk.array["iyc"][0, :, :]
            zc = blk.array["izc"][0, :, :]
            # We want to filter out all eddys that are not possibly going to contribute to this set of ys and zs
            # we are going to refer to the bounding box that encapsulates ALL y,z pairs ad the 'domain' or 'dom'
            patchYmin = yc.min()
            patchYmax = yc.max()
            patchZmin = zc.min()
            patchZmax = zc.max()

            eddysInPatch = np.where(
                (
                    domain.eddyLocs[:, 1] - np.max(domain.sigmas[:, :, 1], axis=1)
                    < patchYmax
                )
                & (
                    domain.eddyLocs[:, 1] + np.max(domain.sigmas[:, :, 1], axis=1)
                    > patchYmin
                )
                & (
                    domain.eddyLocs[:, 2] - np.max(domain.sigmas[:, :, 2], axis=1)
                    < patchZmax
                )
                & (
                    domain.eddyLocs[:, 2] + np.max(domain.sigmas[:, :, 2], axis=1)
                    > patchZmin
                )
            )

            comm.send(yc, dest=sendRank, tag=12)
            comm.send(zc, dest=sendRank, tag=13)
            comm.send(domain.eddy_locs[eddysInPatch], dest=sendRank, tag=18)
            comm.send(domain.eps[eddysInPatch], dest=sendRank, tag=19)
            comm.send(domain.sigmas[eddysInPatch], dest=sendRank, tag=20)

        # Let the zeroth block do some work too
        blk = mb[myBlockNums[i] - 1]
        yc = blk.array["iyc"][0, :, :]
        zc = blk.array["izc"][0, :, :]
    # Other blocks receive for their data
    else:
        yc = comm.recv(source=0, tag=12)
        zc = comm.recv(source=0, tag=13)
        domain.eddy_locs = comm.recv(source=0, tag=18)
        domain.eps = comm.recv(source=0, tag=19)
        domain.sigmas = comm.recv(source=0, tag=20)

    print(f"Rank {rank} is working on patch #{pn}")
    # Now everyone generates their primes like usual
    if progress:
        print(
            "******************************* Generating u' *********************************"
        )
    up, vp, wp = sempy.generate_primes(
        yc,
        zc,
        domain,
        seminp["nframes"],
        normalization=seminp["normalization"],
        interpolate=seminp["interpolate"],
        convect=seminp["convect"],
        shape=seminp["shape"],
        progress=progress,
    )

    # For a periodic spline boundary conditions, the end values must be within machine precision, these
    # end values should already be close, but just in case we set them to identical.
    if seminp["periodicX"]:
        up[-1, :, :] = up[0, :, :]
        vp[-1, :, :] = vp[0, :, :]
        wp[-1, :, :] = wp[0, :, :]
        bc = "periodic"
    else:
        bc = "not-a-knot"

    if seminp["flipdom"]:
        vp = -vp
        wp = -wp

    t = np.linspace(0, seminp["totalTime"], seminp["nframes"])
    # Add the mean profile here
    Upu = domain.ubarInterp(yc) + up
    fu = itrp.CubicSpline(t, Upu, bc_type=bc, axis=0)
    fv = itrp.CubicSpline(t, vp, bc_type=bc, axis=0)
    fw = itrp.CubicSpline(t, wp, bc_type=bc, axis=0)

    for interval in range(seminp["nframes"] - 1):
        fileName = outputDir + "/alphas_{:06d}_{:04d}".format(pn, interval + 1)
        with open(fileName, "w") as f:
            np.save(f, fu.c[:, interval, :, :])
            np.save(f, fv.c[:, interval, :, :])
            np.save(f, fw.c[:, interval, :, :])
