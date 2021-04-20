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
    fx = fxx*fxy*fxz

    #For tracking purposes, see if this point has zero contributions
    tent.empty = np.any( ( np.sum(fx, axis=0) == 0.0 ) )

    return fx

if __name__ == '__main__':

    nsig = 201
    u_length_scale = 1.0
    v_length_scale = 0.5
    w_length_scale = 0.25
    sigs = np.empty((nsig,3))
    line = np.linspace(-u_length_scale*1.5,u_length_scale*1.5,nsig)
    for ii,i in enumerate(line):
        dists = np.array([[i,0,0]])
        sigmas= np.zeros((1,3,3))
        sigmas[:,0,:] = u_length_scale
        sigmas[:,1,:] = v_length_scale
        sigmas[:,2,:] = w_length_scale
        sigs[ii,:] = tent(dists,sigmas)[0]

    import matplotlib.pyplot as plt
    import matplotlib
    if matplotlib.checkdep_usetex(True):
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['figure.figsize'] = (6,4.5)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.size'] = 14
    plt.rcParams['lines.linewidth'] = 1.0

    fig,ax = plt.subplots()
    lo = ax.plot(line,sigs)
    ax.set_title(r'$f(|x-\sigma_{i}|)$')
    ax.set_xlabel(r'$x$')
    plt.legend(lo, (r'$\sigma_{u}$='+f'{u_length_scale}',r'$\sigma_{v}=$'+f'{v_length_scale}',r'$\sigma_{w}=$'+f'{w_length_scale}'))
    plt.grid(linestyle='--')
    plt.show()
