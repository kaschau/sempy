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

'''
import raptorpy as rp
import numpy as np
from scipy.io import FortranFile
from scipy import interpolate as itrp
import os
import sempy
import sys

input_file = sys.argv[1]
##############################
# Read in SEM parameters
##############################
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
        except(ValueError):
            seminp[key] = val
#We need the periodic inputs to be bools
for s in ['periodic_x','periodic_y','periodic_z']:
    if s == 'True':
        seminp[s] = True
    else:
        seminp[s] = False

##############################
# End Modify Sections
##############################

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

#Populate the domain
domain.populate(seminp['C_Eddy'],
                seminp['population_method'],
                convect=seminp['convect'])
#Create the eps
domain.generate_eps()
#Make it periodic
domain.make_periodic(periodic_x=seminp['periodic_x'],
                     periodic_y=seminp['periodic_y'],
                     periodic_z=seminp['periodic_z'])
#Compute sigmas
domain.compute_sigmas()
#Print out a summarry
domain.print_info()

print(f'Generating signal that is {seminp["total_time"]} [s] long, with {seminp["nframes"]} frames.\n')

############################
# RAPTOR
############################
nblks = len([f for f in os.listdir(seminp['grid_path']) if f.startswith('g.')])
mb = rp.multiblock.grid(nblks)
rp.readers.read_raptor_grid(mb,seminp['grid_path'])
inp = rp.readers.read_raptor_input_file(seminp['inp_path'])

patch_num = []
block_num = []
with open('patches.txt','r') as f:
    for line in [i for i in f.readlines() if not i.startswith('#')]:
        l = line.strip().split(',')
        patch_num.append(int(l[0]))
        block_num.append(int(l[1]))

npatches = len(patch_num)

mb.dim_grid(inp)
mb.compute_metrics(xc=False,xu=True,xv=True,xw=True)

for bn,pn in zip(block_num,patch_num):
    blk = mb[bn-1]
    print('\n*******************************************************************************')
    print(f'**************************** Working on Patch #{pn} ******************************')
    print('*******************************************************************************\n')

    print('########## Computing Fluctuations ############')
    face_y = blk.yu[:,:,0].ravel()
    face_z = blk.zu[:,:,0].ravel()

    #We want to store the U momentum points only later, so remember which (y,z) pairs correspond
    # to the U momentum faces.
    ushape = tuple([seminp['nframes']]+list(blk.yu[:,:,0].shape))
    length = face_y.shape[0]

    #We have all the U momentum face centers in the list of points to calculate the fluctuations,
    # now we will add the outer most V,W face centers to the list for interpolation purposes.
    face_y = np.concatenate((face_y, blk.yv[:, 0,0]))
    face_y = np.concatenate((face_y, blk.yv[:,-1,0]))
    face_y = np.concatenate((face_y, blk.yw[ 0,:,0]))
    face_y = np.concatenate((face_y, blk.yw[-1,:,0]))
    face_z = np.concatenate((face_z, blk.zv[:, 0,0]))
    face_z = np.concatenate((face_z, blk.zv[:,-1,0]))
    face_z = np.concatenate((face_z, blk.zw[ 0,:,0]))
    face_z = np.concatenate((face_z, blk.zw[-1,:,0]))

    upp,vpp,wpp = sempy.generate_primes(face_y, face_z, domain,
                                        seminp['nframes'],
                                        normalization=seminp['normalization'])
    # up[ nframe , (y,z) pair]

    #We already have the u fluctuations for ALL the U momentum faces, so we just pull those directly.
    up = upp[:, 0:length].reshape(ushape)

    #We now create interpolators for the v and w fluctuations.
    vshape = tuple([seminp['nframes']]+list(blk.yv[:,:,0].shape))
    vp = np.empty(vshape)
    wshape = tuple([seminp['nframes']]+list(blk.yw[:,:,0].shape))
    wp = np.empty(wshape)
    print('########## Interpolating to v,w ############')
    for i in range(seminp['nframes']):
        vsinterp = itrp.LinearNDInterpolator(np.stack((face_y,face_z),axis=-1), vpp[i,:])
        wsinterp = itrp.LinearNDInterpolator(np.stack((face_y,face_z),axis=-1), wpp[i,:])
        vp[i,:,:] = vsinterp(np.stack((blk.yv[:,:,0].ravel(),blk.zv[:,:,0].ravel()),axis=-1)).reshape(vshape[1:])
        wp[i,:,:] = wsinterp(np.stack((blk.yw[:,:,0].ravel(),blk.zw[:,:,0].ravel()),axis=-1)).reshape(wshape[1:])

    #For a periodic spline boundary conditions, the end values must be within machine precision, these
    # end values should already be close, but just in case we set them to identical.
    if periodic_x:
        up[-1,:,:] = up[0,:,:]
        vp[-1,:,:] = vp[0,:,:]
        wp[-1,:,:] = wp[0,:,:]

    up = up/inp['refvl']['U_ref']
    vp = vp/inp['refvl']['U_ref']
    wp = wp/inp['refvl']['U_ref']

    tme = total_time*inp['refvl']['U_ref']/inp['refvl']['L_ref']
    t = np.linspace(0,tme,seminp['nframes'])
    bc = 'periodic'
    #Add the mean profile here
    Upu = domain.Ubar_interp(blk.yu[:,:,0])/inp['refvl']['U_ref'] + up
    fu = itrp.CubicSpline(t,Upu,bc_type=bc,axis=0)
    fv = itrp.CubicSpline(t,vp,bc_type=bc,axis=0)
    fw = itrp.CubicSpline(t,wp,bc_type=bc,axis=0)

    if not os.path.exists('alphas'):
        os.makedirs('alphas')

    for interval in range(seminp['nframes']-1):
        file_name = './alphas/alphas_{:06d}_{:03d}'.format(pn,interval+1)
        with FortranFile(file_name, 'w') as f90:
            f90.write_record(fu.c[:,interval,:,:].astype(np.float64))
            f90.write_record(fv.c[:,interval,:,:].astype(np.float64))
            f90.write_record(fw.c[:,interval,:,:].astype(np.float64))
