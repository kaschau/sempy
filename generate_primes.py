import numpy as np
np.random.seed(1010)

def generate_primes(ys,zs,domain,nframes,u='u'):

    #check if we have eddys or not
    if not hasattr(domain,'eddy_locs'):
        raise ValueError('Please populate your domain before trying to generate fluctiations')

    #set the index for what fluctuation we want
    if u == 'u':
        k = 0
    elif u == 'v':
        k = 1
    elif u == 'w':
        k = 2

    #Define "time" points for frames
    xs = np.linspace(0,domain.xmax,nframes)
    #Storage for fluctuations
    primes = np.empty((len(ys),nframes))
    print(f'Generating {u} signal:')
    #Loop over each location
    for i,(y,z) in enumerate(zip(ys,zs)):
        print('Generating signal at y=',y)
        #Find eddies that contribute on line for y,z
        eddy_on_line = np.where( (np.abs( domain.eddy_locs[:,1] - y ) < domain.sigmas[:,k,1] )
                               & (np.abs( domain.eddy_locs[:,2] - z ) < domain.sigmas[:,k,2] ) )
        for j,x in enumerate(xs):

            #Find all non zero eddies for u_k'
            x_dist = np.abs( domain.eddy_locs[eddy_on_line][:,0] - x )
            eddy_on_point = np.where( x_dist < domain.sigmas[eddy_on_line][:,k,0] )

            x_dist = np.abs( domain.eddy_locs[eddy_on_line][eddy_on_point][:,0] - x )
            y_dist = np.abs( domain.eddy_locs[eddy_on_line][eddy_on_point][:,1] - y )
            z_dist = np.abs( domain.eddy_locs[eddy_on_line][eddy_on_point][:,2] - z )

            x_sigma = domain.sigmas[eddy_on_line][eddy_on_point][:,k,0]
            y_sigma = domain.sigmas[eddy_on_line][eddy_on_point][:,k,1]
            z_sigma = domain.sigmas[eddy_on_line][eddy_on_point][:,k,2]

            #Individual f(x) 'tent' function
            fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
            fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
            fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

            #Total f(x)
            fx  = np.sqrt(domain.VB)/np.sqrt(np.product((x_sigma,y_sigma,z_sigma),axis=0)) * fxx*fxy*fxz

            A = domain.aij[eddy_on_line][eddy_on_point]
            e = domain.eps_k[eddy_on_line][eddy_on_point]
            ck = np.matmul(A, e).reshape(e.shape[0:2])

            Xk = ck*fx.reshape(fx.shape[0],1)

            prime = 1.0/np.sqrt(domain.neddy) * np.sum( Xk ,axis=0)

            primes[i,j] = prime[k]

    return primes
