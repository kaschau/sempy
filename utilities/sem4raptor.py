import raptorpy as rp
import numpy as np
from scipy.io import FortranFile
from scipy import interpolate as itrp
import os

##############################
# SEM parameters - Modify Here
##############################
import sempy

#Box geom
domain_type = 'channel' #'bl'
y_height = 0.0508
z_width = 0.00544
# Periodicity
periodic_x = True  #periodic in time?
periodic_y = False
periodic_z = True

#Flow - Julia Example
viscosity = 3.33e-5
delta = y_height/2.0
utau = 34.663
Ublk = 895
total_time = 0.002   #time of signal, used to determine length of box

# SEM opitions
nframes = 50
C_Eddy = 3.0 #Eddy Density
sigmas_from = 'jarrin'
stats_from = 'moser'
profile_from = 'channel'
population_method = 'PDF'
normalization = 'exact'

#RAPTOR INFO
nblk = 10
grid_path = './'
inp_path = './dtms.inp'
##############################
# End Modify Sections
##############################

#Initialize domain
domain = sempy.geometries.box(domain_type,
                              Ublk,
                              total_time,
                              y_height,
                              z_width,
                              delta,
                              utau,
                              viscosity)

#Set flow properties from existing data
domain.set_sem_data(sigmas_from=sigmas_from,
                    stats_from=stats_from,
                    profile_from=profile_from,
                    scale_factor=1.0)

#Populate the domain
domain.populate(C_Eddy, population_method)
#Create the eps
domain.generate_eps()
#Make it periodic
domain.make_periodic(periodic_x=periodic_x,
                     periodic_y=periodic_y,
                     periodic_z=periodic_z)
#Compute sigmas
domain.compute_sigmas()
#Print out a summarry
domain.print_info()

print(f'Generating signal that is {total_time} [s] long, with {nframes} frames.\n')

############################
# RAPTOR
############################
mb = rp.multiblock.grid(nblk)
rp.readers.read_raptor_grid(mb,grid_path)
inp = rp.readers.read_raptor_input_file(inp_path)

patch_num = []
block_num = []
face_dir  = []
with open('patches.txt','r') as f:
    for line in [i for i in f.readlines() if not i.startswith('#')]:
        l = line.strip().split(',')
        patch_num.append(int(l[0]))
        block_num.append(int(l[1]))
        face_dir.append(l[2])

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
    ushape = tuple([nframes]+list(blk.yu[:,:,0].shape))
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

    upp,vpp,wpp = sempy.generate_primes(face_y, face_z, domain, nframes, normalization=normalization)
    # up[ nframe , (y,z) pair]

    #We already have the u fluctuations for ALL the U momentum faces, so we just pull those directly.
    up = upp[:, 0:length].reshape(ushape)

    #We now create interpolators for the v and w fluctuations.
    vshape = tuple([nframes]+list(blk.yv[:,:,0].shape))
    vp = np.empty(vshape)
    wshape = tuple([nframes]+list(blk.yw[:,:,0].shape))
    wp = np.empty(wshape)
    print('########## Interpolating to v,w ############')
    for i in range(nframes):
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
    t = np.linspace(0,tme,nframes)
    bc = 'periodic'
    #Add the mean profile here
    Upu = domain.Ubar_interp(blk.yu[:,:,0])/inp['refvl']['U_ref'] + up
    fu = itrp.CubicSpline(t,Upu,bc_type=bc,axis=0)
    fv = itrp.CubicSpline(t,vp,bc_type=bc,axis=0)
    fw = itrp.CubicSpline(t,wp,bc_type=bc,axis=0)

    if not os.path.exists('alphas'):
        os.makedirs('alphas')

    for interval in range(nframes-1):
        file_name = './alphas/alphas_{:06d}_{:03d}'.format(pn,interval+1)
        with FortranFile(file_name, 'w') as f90:
            f90.write_record(fu.c[:,interval,:,:].astype(np.float64))
            f90.write_record(fv.c[:,interval,:,:].astype(np.float64))
            f90.write_record(fw.c[:,interval,:,:].astype(np.float64))
