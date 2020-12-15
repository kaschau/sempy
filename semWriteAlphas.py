import numpy as np
import raptorpy as rp
from scipy.io import FortranFile
from scipy import interpolate as itrp
import matplotlib.pyplot as plt
import patch_reader as pr
# Read in the inlet patches to get the numbers you need to make the
# holder array

npatches = 16
nf = 80 # number of frames
inletPatches = pr.read_patches(npatches, nf, './')

output_dir = './'
for p in inletPatches:
    print('Patch', p.npatch)
    ny = p.y.shape[1]
    nz = p.z.shape[0]

    t = np.linspace(0,16,80)

    bc = 'not-a-knot'
    fu = itrp.CubicSpline(t,p.u,bc_type=bc,axis=0)
    fv = itrp.CubicSpline(t,p.v,bc_type=bc,axis=0)
    fw = itrp.CubicSpline(t,p.w,bc_type=bc,axis=0)

    for frame in range(np.shape(p.u)[0]-1):
        ninterval = frame + 1

        file_name = '{}/alphas_{:06d}_{:03d}'.format(output_dir,p.npatch,ninterval)
        with FortranFile(file_name, 'w') as f90:
            f90.write_record(fu.c[:,frame,:,:].astype(np.float64))
            f90.write_record(fv.c[:,frame,:,:].astype(np.float64))
            f90.write_record(fw.c[:,frame,:,:].astype(np.float64))
