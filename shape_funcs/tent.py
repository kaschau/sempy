import numpy as np

def tent(dists,sigmas):
    '''
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
    '''

    shape = (dists.shape[0],1)

    #Individual f(x) 'tent' functions for each contributing point
    fxx = np.sqrt(1.5)*(1.0-np.abs(np.reshape(dists[:,0],shape))/sigmas[:,:,0])
    fxy = np.sqrt(1.5)*(1.0-np.abs(np.reshape(dists[:,1],shape))/sigmas[:,:,1])
    fxz = np.sqrt(1.5)*(1.0-np.abs(np.reshape(dists[:,2],shape))/sigmas[:,:,2])

    np.clip(fxx,0.0,None,out=fxx)
    np.clip(fxy,0.0,None,out=fxy)
    np.clip(fxz,0.0,None,out=fxz)

    #Total f(x) from each contributing point
    return fxx*fxy*fxz
