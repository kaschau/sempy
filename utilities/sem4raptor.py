#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''This utility generates files full of cubic spline coefficients representing a turbulent signal
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

'''

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
output_dir = input_file.split('.')[0]+'_alphas'
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
    with open(input_file,'r') as f:
        for line in [i for i in f.readlines() if not i.replace(' ','').startswith('#') and i.strip() != '']:
            nocomment = line.strip().split('#')[0]
            key,val = tuple(nocomment.replace(' ','').split('='))
            try: #convert numbers to floats or ints
                if '.' in val:
                    seminp[key] = float(val)
                else:
                    seminp[key] = int(val)
            except ValueError:
                seminp[key] = val
        #We need the periodic inputs to be bools
        for s in ['periodic_x','periodic_y','periodic_z']:
            if seminp[s] == 'True':
                seminp[s] = True
            else:
                seminp[s] = False
else:
    seminp = None
seminp = comm.bcast(seminp,root=0)

############################################################################################
# Create the domain based on above inputs
############################################################################################
#Initialize domain
domain = sempy.geometries.box(seminp['domain_type'],
                              seminp['Ublk'],
                              seminp['total_time'],
                              seminp['y_height'],
                              seminp['z_width'],
                              seminp['delta'],
                              seminp['utau'],
                              seminp['viscosity'])

#Set flow properties from existing data
domain.set_sem_data(sigmas_from=seminp['sigmas_from'],
                    stats_from=seminp['stats_from'],
                    profile_from=seminp['profile_from'],
                    scale_factor=1.0)


# Only the zeroth rank populates the domain
if rank == 0:
    #Populate the domain
    domain.populate(seminp['C_Eddy'],
                    seminp['population_method'])
    #Create the eps
    domain.generate_eps()
    #Compute sigmas
    domain.compute_sigmas()
    #Make it periodic
    domain.make_periodic(periodic_x=seminp['periodic_x'],
                         periodic_y=seminp['periodic_y'],
                         periodic_z=seminp['periodic_z'])

############################################################################################
# Read in patches.txt
############################################################################################
# Only the zeroth rank reads in patches.txt
if rank == 0:
    patch_num = []
    block_num = []
    with open('patches.txt','r') as f:
        for line in [i for i in f.readlines() if not i.startswith('#')]:
            l = line.strip().split(',')
            patch_num.append(int(l[0]))
            block_num.append(int(l[1]))

    npatches = len(patch_num)
    # Now we assign patches to ranks
    max_ranks_per_block = int(npatches/size+npatches%size)
    all_rank_patches = [[None for j in range(max_ranks_per_block)] for i in range(size)]
    all_rank_blocks = [[None for j in range(max_ranks_per_block)] for i in range(size)]
    i = 0
    j = 0
    for bn,pn in zip(block_num,patch_num):
        all_rank_patches[i][j] = pn
        all_rank_blocks[i][j] = bn
        i += 1
        if i == size:
            i = 0
            j += 1
    my_patch_nums = all_rank_patches[0]
    my_block_nums = all_rank_blocks[0]
    #Send the list of patch numbers to the respective ranks ranks
    for i,rsp in enumerate(all_rank_patches[1::]):
        comm.send(rsp, dest=i+1, tag=11)
else:
    my_patch_nums = comm.recv(source=0, tag=11)


############################################################################################
# RAPTOR stuff (read in grid and make the patches)
############################################################################################
# Only rank ones reads in the grid
if rank == 0:
    nblks = len([f for f in os.listdir(seminp['grid_path']) if f.startswith('g.')])
    mb = rp.multiblock.grid(nblks)
    rp.readers.read_raptor_grid(mb,seminp['grid_path'])
    rpinp = rp.readers.read_raptor_input_file(seminp['inp_path'])
    mb.dim_grid(rpinp)
    mb.compute_metrics(xc=False,xu=True,xv=True,xw=True)
else:
    rpinp = None
# Send the raptor input file to everyone
rpinp = comm.bcast(rpinp,root=0)

# Progress bar gets messsy with all the blocks, so we only show progress with rank 0
if rank == 0:
    progress=True
else:
    progress=False

if progress:
    #Print out a summarry
    domain.print_info()
    print(f'Generating signal that is {seminp["total_time"]} [s] long, with {seminp["nframes"]} frames.\n')
    print('\n*******************************************************************************')
    print('**************************** Generating Primes ********************************')
    print('*******************************************************************************\n')

for i,pn in enumerate(my_patch_nums):
    #If we are the zeroth block, then we compute and send the patches to all the other blocks
    if rank == 0:
        for send_rank in [j+1 for j in range(size-1)]:
            send_patch_num = all_rank_patches[send_rank][i]
            send_block_num = all_rank_blocks[send_rank][i]
            blk = mb[send_block_num-1]
            #Create the patch for this block
            yu = blk.yu[:,:,0]
            zu = blk.zu[:,:,0]
            yv = blk.yv[:,:,0]
            zv = blk.zv[:,:,0]
            yw = blk.yw[:,:,0]
            zw = blk.zw[:,:,0]
            #We want to filter out all eddys that are not possibly going to contribute to this set of ys and zs
            #we are going to refer to the bounding box that encapsulates ALL y,z pairs ad the 'domain' or 'dom'
            patch_ymin = min([yu.min(),yv.min(),yw.min()])
            patch_ymax = max([yu.max(),yv.max(),yw.max()])
            patch_zmin = min([zu.min(),zv.min(),zw.min()])
            patch_zmax = max([zu.max(),zv.max(),zw.max()])

            eddys_in_patch = np.where( ( domain.eddy_locs[:,1] - np.max(domain.sigmas[:,:,1], axis=1) < patch_ymax )
                                   & ( domain.eddy_locs[:,1] + np.max(domain.sigmas[:,:,1], axis=1) > patch_ymin )
                                   & ( domain.eddy_locs[:,2] - np.max(domain.sigmas[:,:,2], axis=1) < patch_zmax )
                                   & ( domain.eddy_locs[:,2] + np.max(domain.sigmas[:,:,2], axis=1) > patch_zmin ) )

            comm.send(yu, dest=send_rank, tag=12)
            comm.send(zu, dest=send_rank, tag=13)
            comm.send(yv, dest=send_rank, tag=14)
            comm.send(zv, dest=send_rank, tag=15)
            comm.send(yw, dest=send_rank, tag=16)
            comm.send(zw, dest=send_rank, tag=17)
            comm.send(domain.eddy_locs[eddys_in_patch], dest=send_rank, tag=18)
            comm.send(domain.eps[eddys_in_patch], dest=send_rank, tag=19)
            comm.send(domain.sigmas[eddys_in_patch], dest=send_rank, tag=20)

        #Let the zeroth block do some work too
        blk = mb[my_block_nums[i]-1]
        yu = blk.yu[:,:,0]
        zu = blk.zu[:,:,0]
        yv = blk.yv[:,:,0]
        zv = blk.zv[:,:,0]
        yw = blk.yw[:,:,0]
        zw = blk.zw[:,:,0]
    # Other blocks receive for their data
    else:
        yu = comm.recv(source=0, tag=12)
        zu = comm.recv(source=0, tag=13)
        yv = comm.recv(source=0, tag=14)
        zv = comm.recv(source=0, tag=15)
        yw = comm.recv(source=0, tag=16)
        zw = comm.recv(source=0, tag=17)
        domain.eddy_locs = comm.recv(source=0, tag=18)
        domain.eps = comm.recv(source=0, tag=19)
        domain.sigmas = comm.recv(source=0, tag=20)

    print(f'Rank {rank} is working on patch #{pn}')
    #Now everyone generates their primes like usual
    face_y = yu.ravel()
    face_z = zu.ravel()

    #We want to store the U momentum points only later, so remember which (y,z) pairs correspond
    # to the U momentum faces.
    ushape = tuple([seminp['nframes']]+list(yu.shape))
    length = face_y.shape[0]

    #We have all the U momentum face centers in the list of points to calculate the fluctuations,
    # now we will add the outer most V,W face centers to the list for interpolation purposes.
    face_y = np.concatenate((face_y, yv[:, 0]))
    face_y = np.concatenate((face_y, yv[:,-1]))
    face_y = np.concatenate((face_y, yw[ 0,:]))
    face_y = np.concatenate((face_y, yw[-1,:]))
    face_z = np.concatenate((face_z, zv[:, 0]))
    face_z = np.concatenate((face_z, zv[:,-1]))
    face_z = np.concatenate((face_z, zw[ 0,:]))
    face_z = np.concatenate((face_z, zw[-1,:]))

    upp,vpp,wpp = sempy.generate_primes(face_y, face_z, domain,
                                        seminp['nframes'],
                                        normalization=seminp['normalization'],
                                        progress=progress)
    # upp[ nframe , (y,z) pair]

    #We already have the u fluctuations for ALL the U momentum faces, so we just pull those directly.
    up = upp[:, 0:length].reshape(ushape)

    #We now create interpolators for the v and w fluctuations.
    vshape = tuple([seminp['nframes']]+list(yv.shape))
    vp = np.empty(vshape)
    wshape = tuple([seminp['nframes']]+list(yw.shape))
    wp = np.empty(wshape)

    for i in range(seminp['nframes']):
        vsinterp = itrp.LinearNDInterpolator(np.stack((face_y,face_z),axis=-1), vpp[i,:])
        wsinterp = itrp.LinearNDInterpolator(np.stack((face_y,face_z),axis=-1), wpp[i,:])
        vp[i,:,:] = vsinterp(np.stack((yv.ravel(),zv.ravel()),axis=-1)).reshape(vshape[1:])
        wp[i,:,:] = wsinterp(np.stack((yw.ravel(),zw.ravel()),axis=-1)).reshape(wshape[1:])

    #For a periodic spline boundary conditions, the end values must be within machine precision, these
    # end values should already be close, but just in case we set them to identical.
    if seminp['periodic_x']:
        up[-1,:,:] = up[0,:,:]
        vp[-1,:,:] = vp[0,:,:]
        wp[-1,:,:] = wp[0,:,:]
        bc = 'periodic'
    else:
        bc = 'not-a-knot'

    up = up/rpinp['refvl']['U_ref']
    vp = vp/rpinp['refvl']['U_ref']
    wp = wp/rpinp['refvl']['U_ref']

    tme = seminp['total_time']*rpinp['refvl']['U_ref']/rpinp['refvl']['L_ref']
    t = np.linspace(0,tme,seminp['nframes'])
    #Add the mean profile here
    Upu = domain.Ubar_interp(yu)/rpinp['refvl']['U_ref'] + up
    fu = itrp.CubicSpline(t,Upu,bc_type=bc,axis=0)
    fv = itrp.CubicSpline(t,vp,bc_type=bc,axis=0)
    fw = itrp.CubicSpline(t,wp,bc_type=bc,axis=0)

    for interval in range(seminp['nframes']-1):
        file_name = output_dir+'/alphas_{:06d}_{:03d}'.format(pn,interval+1)
        with FortranFile(file_name, 'w') as f90:
            f90.write_record(fu.c[:,interval,:,:].astype(np.float64))
            f90.write_record(fv.c[:,interval,:,:].astype(np.float64))
            f90.write_record(fw.c[:,interval,:,:].astype(np.float64))
