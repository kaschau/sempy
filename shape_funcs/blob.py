import numpy as np

def blob(dists,sigmas):
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

    fxx = np.ones((dists.shape[0],3))
    for i,(d,s) in enumerate(zip(dists,sigmas)):
        fx = np.zeros((3,3))
        #Loop over each x,y,z direction, find the largest length scale in that direction,
        # compute the "big" eddy, and then the other two small eddys
        for xyz in range(3):
            max_component = np.argmax(s[:,xyz])
            big_eddy = 0.5*(np.cos(np.pi*d[xyz]/s[max_component,xyz])+1.0)

            fx[max_component,xyz] = big_eddy

            smaller_components = [0,1,2]
            smaller_components.remove(max_component)
            for smlr in smaller_components:
                small_eddy = big_eddy*np.cos(np.pi*d[xyz]/(2.0*s[smlr,xyz]))
                fx[smlr,xyz] = small_eddy

        fxx[i,:] = np.product(fx,axis=1)

    #Total f(x) from each contributing eddy
    return fxx

if __name__ == '__main__':

    nsig = 100
    u_length_scale = 1.0
    v_length_scale = 0.5
    w_length_scale = 0.25
    sigs = np.empty((nsig,3))
    line = np.linspace(-u_length_scale,u_length_scale,nsig)
    for ii,i in enumerate(line):
        dists = np.array([[i,0,0]])
        sigmas= np.zeros((1,3,3))
        sigmas[:,0,:] = u_length_scale
        sigmas[:,1,:] = v_length_scale
        sigmas[:,2,:] = w_length_scale
        sigs[ii,:] = blob(dists,sigmas)[0]

    import matplotlib.pyplot as plt
    lo = plt.plot(line,sigs)
    plt.legend(lo, (f'u={u_length_scale}', f'v={v_length_scale}', f'w={w_length_scale}'))
    plt.grid()
    plt.show()
