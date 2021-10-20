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

import raptorpy as rp
import sempy
import numpy as np
from mpi4py import MPI
from scipy.io import FortranFile
from scipy import interpolate as itrp
import os
import sys

input_file = sys.argv[1]

# Initialize the parallel information for each rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Use the input file name as the output directory
output_dir = input_file.split(".")[0] + "_alphas"
if rank == 0:
    if os.path.exists(output_dir):
        pass
    else:
        os.makedirs(output_dir)
############################################################################################
# Read in SEM parameters
############################################################################################
# The zeroth rank will read in the input file and give it to the other ranks
if rank == 0:
    seminp = dict()
    with open(input_file, "r") as f:
        for line in [
            i
            for i in f.readlines()
            if not i.replace(" ", "").startswith("#") and i.strip() != ""
        ]:
            nocomment = line.strip().split("#")[0]
            key, val = tuple(nocomment.replace(" ", "").split("="))
            try:  # convert numbers to floats or ints
                if "." in val:
                    seminp[key] = float(val)
                else:
                    seminp[key] = int(val)
            except ValueError:
                if val in ["True", "true", "t", "T"]:
                    seminp[key] = True
                elif val in ["False", "true", "f", "F"]:
                    seminp[key] = False
                else:
                    seminp[key] = val

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
    seminp["domain_type"],
    seminp["Ublk"],
    seminp["total_time"],
    seminp["y_height"],
    seminp["z_width"],
    seminp["delta"],
    seminp["utau"],
    seminp["viscosity"],
)

# Set flow properties from existing data
domain.set_sem_data(
    sigmas_from=seminp["sigmas_from"],
    stats_from=seminp["stats_from"],
    profile_from=seminp["profile_from"],
    scale_factor=seminp["scale_factor"],
)

# Only the zeroth rank populates the domain
if rank == 0:
    # Populate the domain
    domain.populate(seminp["C_Eddy"], seminp["population_method"])
    # Create the eps
    domain.generate_eps()
    # Compute sigmas
    domain.compute_sigmas()
    # Make it periodic
    domain.make_periodic(
        periodic_x=seminp["periodic_x"],
        periodic_y=seminp["periodic_y"],
        periodic_z=seminp["periodic_z"],
    )
    temp_neddy = domain.neddy
else:
    temp_neddy = None
# We have to overwrite the worker processes' neddy property
# so that each worker know how many eddys are in the actual
# domain
temp_neddy = comm.bcast(temp_neddy, root=0)
if rank != 0:
    domain._neddy = temp_neddy

############################################################################################
# Read in patches.txt
############################################################################################
# Only the zeroth rank reads in patches.txt
if rank == 0:
    patch_num = []
    block_num = []
    with open("patches.txt", "r") as f:
        for line in [i for i in f.readlines() if not i.startswith("#")]:
            li = line.strip().split(",")
            patch_num.append(int(li[0]))
            block_num.append(int(li[1]))

    npatches = len(patch_num)
    # Now we assign patches to ranks
    if size > npatches:
        print(
            "\n\nFYI, patches are assigned to processors, so using more processors than patches gives you no performace increase.\n\n"
        )
        max_ranks_per_block = 1
    else:
        max_ranks_per_block = int(npatches / size) + (1 if npatches % size > 0 else 0)
    all_rank_patches = [[None for j in range(max_ranks_per_block)] for i in range(size)]
    all_rank_blocks = [[None for j in range(max_ranks_per_block)] for i in range(size)]
    i = 0
    j = 0
    for bn, pn in zip(block_num, patch_num):
        all_rank_patches[i][j] = pn
        all_rank_blocks[i][j] = bn
        i += 1
        if i == size:
            i = 0
            j += 1
    my_patch_nums = all_rank_patches[0]
    my_block_nums = all_rank_blocks[0]
    # Send the list of patch numbers to the respective ranks ranks
    for i, rsp in enumerate(all_rank_patches[1::]):
        comm.send(rsp, dest=i + 1, tag=11)
else:
    my_patch_nums = comm.recv(source=0, tag=11)


############################################################################################
# RAPTOR stuff (read in grid and make the patches)
############################################################################################
# Only rank ones reads in the grid
if rank == 0:
    nblks = len([f for f in os.listdir(seminp["grid_path"]) if f.startswith("g.")])
    if nblks == 0:
        nblks = len([f for f in os.listdir(seminp["grid_path"]) if f == "grid"])
    if nblks == 0:
        raise ValueError(f'Cant find any grid files in {seminp["grid_path"]}')

    mb = rp.multiblock.grid(nblks)
    rp.readers.read_raptor_grid(mb, seminp["grid_path"])
    rpinp = rp.readers.read_raptor_input_file(seminp["inp_path"])
    mb.dim_grid(rpinp)
    # Flip the grid if it is oriented upside down in the true grid
    if seminp["flipdom"]:
        for bn in block_num:
            blk = mb[bn - 1]
            blk.y = -blk.y
            blk.z = -blk.z
    # Determinw the extents the grid needs to be shifted
    ymin = np.inf
    zmin = np.inf
    for bn in block_num:
        blk = mb[bn - 1]
        ymin = min(ymin, blk.y[:, :, 0].min())
        zmin = min(zmin, blk.z[:, :, 0].min())
    # Now shift the grid to match the domain (0,0) starting point
    for bn in block_num:
        blk = mb[bn - 1]
        blk.y = blk.y - ymin
        blk.z = blk.z - zmin
        # Compute the locations of face centers
        blk.compute_U_face_centers()
        blk.compute_V_face_centers()
        blk.compute_W_face_centers()

else:
    rpinp = None
# Send the raptor input file to everyone
rpinp = comm.bcast(rpinp, root=0)

# Progress bar gets messsy with all the blocks, so we only show progress with rank 0
if rank == 0:
    progress = True
else:
    progress = False

if progress:
    # Print out a summarry
    domain.print_info()
    print(
        f'Generating signal that is {seminp["total_time"]} [s] long, with {seminp["nframes"]} frames using {seminp["convect"]} convection speed.\n'
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

for i, pn in enumerate(my_patch_nums):
    if rank != 0 and pn is None:
        continue
    # If we are the zeroth block, then we compute and send the patches to all the other blocks
    if rank == 0:
        for send_rank in [j + 1 for j in range(size - 1)]:
            send_patch_num = all_rank_patches[send_rank][i]
            send_block_num = all_rank_blocks[send_rank][i]
            if send_patch_num is None:
                continue
            blk = mb[send_block_num - 1]
            # Create the patch for this block
            yu = blk.yu[:, :, 0]
            zu = blk.zu[:, :, 0]
            # We want to filter out all eddys that are not possibly going to contribute to this set of ys and zs
            # we are going to refer to the bounding box that encapsulates ALL y,z pairs ad the 'domain' or 'dom'
            patch_ymin = yu.min()
            patch_ymax = yu.max()
            patch_zmin = zu.min()
            patch_zmax = zu.max()

            eddys_in_patch = np.where(
                (
                    domain.eddy_locs[:, 1] - np.max(domain.sigmas[:, :, 1], axis=1)
                    < patch_ymax
                )
                & (
                    domain.eddy_locs[:, 1] + np.max(domain.sigmas[:, :, 1], axis=1)
                    > patch_ymin
                )
                & (
                    domain.eddy_locs[:, 2] - np.max(domain.sigmas[:, :, 2], axis=1)
                    < patch_zmax
                )
                & (
                    domain.eddy_locs[:, 2] + np.max(domain.sigmas[:, :, 2], axis=1)
                    > patch_zmin
                )
            )

            comm.send(yu, dest=send_rank, tag=12)
            comm.send(zu, dest=send_rank, tag=13)
            comm.send(domain.eddy_locs[eddys_in_patch], dest=send_rank, tag=18)
            comm.send(domain.eps[eddys_in_patch], dest=send_rank, tag=19)
            comm.send(domain.sigmas[eddys_in_patch], dest=send_rank, tag=20)

        # Let the zeroth block do some work too
        blk = mb[my_block_nums[i] - 1]
        yu = blk.yu[:, :, 0]
        zu = blk.zu[:, :, 0]
    # Other blocks receive for their data
    else:
        yu = comm.recv(source=0, tag=12)
        zu = comm.recv(source=0, tag=13)
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
        yu,
        zu,
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
    if seminp["periodic_x"]:
        up[-1, :, :] = up[0, :, :]
        vp[-1, :, :] = vp[0, :, :]
        wp[-1, :, :] = wp[0, :, :]
        bc = "periodic"
    else:
        bc = "not-a-knot"

    up = up / rpinp["refvl"]["U_ref"]
    vp = vp / rpinp["refvl"]["U_ref"]
    wp = wp / rpinp["refvl"]["U_ref"]

    if seminp["flipdom"]:
        vp = -vp
        wp = -wp

    tme = seminp["total_time"] * rpinp["refvl"]["U_ref"] / rpinp["refvl"]["L_ref"]
    t = np.linspace(0, tme, seminp["nframes"])
    # Add the mean profile here
    Upu = domain.Ubar_interp(yu) / rpinp["refvl"]["U_ref"] + up
    fu = itrp.CubicSpline(t, Upu, bc_type=bc, axis=0)
    fv = itrp.CubicSpline(t, vp, bc_type=bc, axis=0)
    fw = itrp.CubicSpline(t, wp, bc_type=bc, axis=0)

    for interval in range(seminp["nframes"] - 1):
        file_name = output_dir + "/alphas_{:06d}_{:04d}".format(pn, interval + 1)
        with FortranFile(file_name, "w") as f90:
            f90.write_record(fu.c[:, interval, :, :].astype(np.float64))
            f90.write_record(fv.c[:, interval, :, :].astype(np.float64))
            f90.write_record(fw.c[:, interval, :, :].astype(np.float64))
