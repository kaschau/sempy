"""This utility generates files full of cubic spline coefficients representing
a turbulent signal that can be applied at the inlet of a PEREGRINE simulation.
Just call this utility with an input file.

The inlet faces are found based on the bcFam setting in conn.yaml corrseponding
to the name of the input file, i.e. an input file called synthTurb.yaml will
look for bcFam tagged synthTurb in conn.yaml.

Example
-------
/path/to/sempy/utilities/semForPEREGRINE.py <sem.yaml>

The output is a directory with the name of your input file, +'Alphas' full of
your alpha coeffs. Each inlet face gets one file with all the frames.

You can also run this utility in parallel, just use mpiexec

Example
-------
mpiexec -np <np> /path/to/sempy/utilities/semForPEREGRINE.py <sem.yaml>

"""

import os
import sys

import numpy as np
import peregrinepy as pg
import sempy
import yaml
from mpi4py import MPI
from scipy import interpolate as itrp


def getSlice(fn):
    if fn == 1:
        return np.s_[0, :, :]
    elif fn == 2:
        return np.s_[-1, :, :]
    elif fn == 3:
        return np.s_[:, 0, :]
    elif fn == 4:
        return np.s_[:, -1, :]
    elif fn == 5:
        return np.s_[:, :, 0]
    elif fn == 6:
        return np.s_[:, :, -1]


inputFile = sys.argv[1]
bcFam = inputFile.split(".")[0]

# Initialize the parallel information for each rank
MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Use the input file name as the output directory
outputDir = bcFam + "Alphas"
if rank == 0:
    if os.path.exists(outputDir):
        pass
    else:
        os.makedirs(outputDir)
###############################################################################
# Read in SEM parameters
###############################################################################
with open(inputFile, "r") as f:
    seminp = yaml.load(f, Loader=yaml.FullLoader)

###############################################################################
# Create the domain based on above inputs
###############################################################################
# Initialize domain
domain = sempy.geometries.box(
    seminp["domainType"],
    seminp["Uo"],
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

###############################################################################
# PEREGRINE stuff (read in grid and make the patches)
###############################################################################
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
    pg.readers.readGrid(mb, seminp["gridPath"])
    pg.readers.readConnectivity(mb, seminp["connPath"])
    # Get the blocks and faces for this inlet
    inletBlocks = []
    inletFaces = []
    for blk in mb:
        for face in blk.faces:
            if face.bcFam == bcFam:
                inletBlocks.append(blk.nblki)
                inletFaces.append(face.nface)
    print(f"\nFound {len(inletBlocks)} blocks for this inlet.")

    # Flip the grid if it is oriented upside down in the true grid
    if seminp["flipdom"]:
        for bn in inletBlocks:
            blk = mb.getBlock(bn)
            blk.array["y"] = -blk.array["y"]
            blk.array["z"] = -blk.array["z"]
    # Determine the extents the grid needs to be shifted
    ymin = np.inf
    zmin = np.inf
    for bn, fn in zip(inletBlocks, inletFaces):
        blk = mb.getBlock(bn)
        s1_ = getSlice(fn)
        ymin = min(ymin, blk.array["y"][s1_].min())
        zmin = min(zmin, blk.array["z"][s1_].min())
    # Now shift the grid to match the domain (0,0) starting point
    for bn in inletBlocks:
        blk = mb.getBlock(bn)
        blk.array["y"] = blk.array["y"] - ymin
        blk.array["z"] = blk.array["z"] - zmin
        # Compute the locations of face centers
        blk.computeMetrics(fdOrder=2)

###############################################################################
# Assign faces to MPI ranks
###############################################################################
# Only the zeroth rank reads in patches.txt
if rank == 0:
    npatches = len(inletFaces)
    # Now we assign patches to ranks
    if size > npatches:
        print(
            "\n\nFYI, patches are assigned to processors, so using more\n",
            "processors than patches gives you no performace increase.\n\n",
        )
        maxRanksPerBlock = 1
    else:
        maxRanksPerBlock = int(npatches / size) + (1 if npatches % size > 0 else 0)
    allRankFaces = [[None for j in range(maxRanksPerBlock)] for i in range(size)]
    allRankBlocks = [[None for j in range(maxRanksPerBlock)] for i in range(size)]
    i = 0
    j = 0
    for bn, fn in zip(inletBlocks, inletFaces):
        allRankFaces[i][j] = fn
        allRankBlocks[i][j] = bn
        i += 1
        if i == size:
            i = 0
            j += 1
    myFaceNums = allRankFaces[0]
    myBlockNums = allRankBlocks[0]
    # Send the list of block and face numbers to the respective ranks
    for i, (rsb, rsf) in enumerate(zip(allRankBlocks[1::], allRankFaces[1::])):
        comm.send(rsb, dest=i + 1, tag=11)
        comm.send(rsf, dest=i + 1, tag=12)
else:
    myBlockNums = comm.recv(source=0, tag=11)
    myFaceNums = comm.recv(source=0, tag=12)
# Progress bar gets messsy with all the blocks,
# so we only show progress with rank 0
if rank == 0:
    progress = True
else:
    progress = False

if progress:
    # Print out a summarry
    print(domain)
    print(
        f'Generating signal that is {seminp["totalTime"]} [s] long,\n',
        f'with {seminp["nframes"]} frames\n',
        f'using {seminp["convect"]} convection speed.\n',
    )
    print(
        " ***********************************\n",
        "******* Generating Primes *********\n",
        "***********************************\n",
    )

for i, (bn, fn) in enumerate(zip(myBlockNums, myFaceNums)):
    if rank != 0 and fn is None:
        continue
    # If we are the zeroth block, then we compute and
    # send the patches to all the other blocks
    if rank == 0:
        for sendRank in [j + 1 for j in range(size - 1)]:
            sendFaceNum = allRankFaces[sendRank][i]
            sendBlockNum = allRankBlocks[sendRank][i]
            if sendFaceNum is None:
                continue
            blk = mb.getBlock(sendBlockNum)
            # Create the patch for this block
            yc = blk.array["iyc"][0, :, :]
            zc = blk.array["izc"][0, :, :]
            # We want to filter out all eddys that are not
            # possibly going to contribute to this set of ys and zs
            # we are going to refer to the bounding box that
            # encapsulates ALL y,z pairs ad the 'domain' or 'dom'
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
            comm.send(domain.eddyLocs[eddysInPatch], dest=sendRank, tag=18)
            comm.send(domain.eps[eddysInPatch], dest=sendRank, tag=19)
            comm.send(domain.sigmas[eddysInPatch], dest=sendRank, tag=20)

        # Let the zeroth block do some work too
        blk = mb.getBlock(myBlockNums[i])
        yc = blk.array["iyc"][0, :, :]
        zc = blk.array["izc"][0, :, :]
    # Other blocks receive for their data
    else:
        yc = comm.recv(source=0, tag=12)
        zc = comm.recv(source=0, tag=13)
        domain.eddyLocs = comm.recv(source=0, tag=18)
        domain.eps = comm.recv(source=0, tag=19)
        domain.sigmas = comm.recv(source=0, tag=20)

    print(f"Rank {rank} is working on Block {bn}, Face {fn}")
    # Now everyone generates their primes like usual

    up, vp, wp = sempy.generatePrimes(
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

    # For a periodic spline boundary conditions, the end values must be within
    # machine precision, these end values should already be close, but just in
    # case we set them to identical.
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

    fileName = outputDir + "/alphas_{}_{}.npy".format(bn, fn)
    with open(fileName, "wb") as f:
        #  shape =      4  nframes  ny nz
        np.save(f, fu.c[:, :, :, :])
        np.save(f, fv.c[:, :, :, :])
        np.save(f, fw.c[:, :, :, :])

# Finalize MPI
MPI.Finalize()
