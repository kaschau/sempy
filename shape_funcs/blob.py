import numpy as np
from numba import njit

@njit(fastmath=True)
def blob(dists,sigmas):
    '''
    Produce an eddy with single volume, and 9 characteristic length scales.

    NOTE: distances in dists are not absolute valued.
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

    fxx = np.ones((dists.shape[0],3))
    for i,(d,s) in enumerate(zip(dists,sigmas)):
        if np.max(np.abs(d)) > np.max(s):
            fxx[i,:] = 0.0
            continue
        fx = np.empty((3,3))

        #Loop over each x,y,z direction, find the largest length scale in that direction,
        # compute the "big" eddy, and then the other two small eddys
        for xyz in range(3):
            if np.abs(d[xyz]) > np.max(s[:,xyz]):
                fx[:,xyz] = 0.0
                break

            max_component = np.argmax(s[:,xyz])
            big_eddy = np.cos(np.pi*d[xyz]/(2.0*s[max_component,xyz]))

            fx[max_component,xyz] = big_eddy

            smaller_components = [0,1,2]
            smaller_components.remove(max_component)
            for smlr in smaller_components:
                small_eddy = big_eddy*np.cos(np.pi*d[xyz]/(2.0*s[smlr,xyz]))
                fx[smlr,xyz] = small_eddy

        for k in range(3):
            fxx[i,k] = np.prod(fx[k])

    #Total f(x) from each contributing eddy
    return fxx

if __name__ == '__main__':

    nsig = 201
    ux_length_scale = 1.0
    uy_length_scale = 0.75
    vx_length_scale = 0.5
    vy_length_scale = 0.25
    wx_length_scale = 0.25
    wy_length_scale = 0.25

    line = np.linspace(-ux_length_scale*1.5,ux_length_scale*1.5,nsig)

    dists = np.zeros((nsig,3))
    dists[:,0] = line

    sigmas= np.ones((nsig,3,3))
    sigmas[:,0,0] = ux_length_scale
    sigmas[:,1,0] = vx_length_scale
    sigmas[:,2,0] = wx_length_scale

    sigs = blob(dists,sigmas)

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
    plt.rcParams['image.cmap'] = 'coolwarm'

    fig,ax = plt.subplots()
    lo = ax.plot(line,sigs)
    ax.set_title(r'$f(|x-\sigma_{i}|)$')
    ax.set_xlabel(r'$x$')
    plt.legend(lo, (r'$\sigma_{u}$='+f'{ux_length_scale}',r'$\sigma_{v}=$'+f'{vx_length_scale}',r'$\sigma_{w}=$'+f'{wx_length_scale}'))
    plt.grid(linestyle='--')
    plt.show()

    ##############################################################
    # Optional 2D plots
    ##############################################################
    plot_2D = True
    if plot_2D:
        X,Y = np.meshgrid(line,line)
        npts = X.ravel().shape[0]

        dists = np.zeros((npts,3))
        dists[:,0] = X.ravel()
        dists[:,1] = Y.ravel()

        sigmas= np.ones((npts,3,3))
        sigmas[:,0,0] = ux_length_scale
        sigmas[:,0,1] = uy_length_scale
        sigmas[:,1,0] = vx_length_scale
        sigmas[:,1,1] = vy_length_scale
        sigmas[:,2,0] = wx_length_scale
        sigmas[:,2,1] = wy_length_scale

        sigs = blob(dists,sigmas)
        fig,ax = plt.subplots()
        tcf = ax.tricontourf(X.ravel(),Y.ravel(),sigs[:,0], levels=np.linspace(-1,1,31))
        fig.colorbar(tcf,ticks=np.linspace(-1,1,11))
        ax.set_title(r'$\sigma_{ux}=$' + f'{ux_length_scale}, '+r'$\sigma_{uy}=$'+f'{uy_length_scale}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(linestyle='--')
        plt.show()

        fig,ax = plt.subplots()
        tcf = ax.tricontourf(X.ravel(),Y.ravel(),sigs[:,1], levels=np.linspace(-1,1,31))
        cb = fig.colorbar(tcf,ticks=np.linspace(-1,1,11))
        ax.set_title(r'$\sigma_{vx}$='+f'{vx_length_scale}, ' + r'$\sigma_{vy}=$'+f'{vy_length_scale}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(linestyle='--')
        plt.show()

        fig,ax = plt.subplots()
        tcf = ax.tricontourf(X.ravel(),Y.ravel(),sigs[:,2], levels=np.linspace(-1,1,31))
        cb = fig.colorbar(tcf,ticks=np.linspace(-1,1,11))
        ax.set_title(r'$\sigma_{wx}$='+f'{wx_length_scale}, ' + r'$\sigma_{wy}=$'+f'{wy_length_scale}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(linestyle='--')
        plt.show()
