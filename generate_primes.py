import numpy as np
import sys
np.random.seed(1010)

def generate_primes(ys,zs,domain,nframes,normalization='exact'):

    #check if we have eddys or not
    if not hasattr(domain,'eddy_locs'):
        raise ValueError('Please populate your domain before trying to generate fluctiations')

    #We want to filter out all eddys that are not possibly going to contribute to this set of ys and zs
    dom_ymin = ys.min()
    dom_ymax = ys.max()
    dom_zmin = zs.min()
    dom_zmax = zs.max()
    eddy_in_dom = dict()
    for k,u in zip(range(3),['u','v','w']):
        eddy_in_dom[u] = np.where( ( domain.eddy_locs[:,1] < dom_ymax + domain.sigmas[:,k,1].max() )
                                 & ( domain.eddy_locs[:,1] > dom_ymin - domain.sigmas[:,k,1].max() )
                                 & ( domain.eddy_locs[:,2] < dom_zmax + domain.sigmas[:,k,2].max() )
                                 & ( domain.eddy_locs[:,2] > dom_zmin - domain.sigmas[:,k,2].max() ) )

    #Define "time" points for frames
    xs = np.linspace(0,domain.x_length,nframes)
    #Storage for fluctuations
    primes = np.empty((len(ys),nframes,3))

    #Loop over each location
    for i,(y,z) in enumerate(zip(ys,zs)):

        #Compute Rij for y location
        Rij = domain.Rij_interp(y)
        #Cholesky decomp of stats
        L = np.linalg.cholesky(Rij)

        print('   y=',y)
        #Find eddies that contribute on line for y,z
        eddy_on_line = dict()
        for k,u in zip(range(3),['u','v','w']):
            eddy_on_line[u] = np.where( (np.abs( domain.eddy_locs[eddy_in_dom[u]][:,1] - y ) < domain.sigmas[eddy_in_dom[u]][:,k,1] )
                                      & (np.abs( domain.eddy_locs[eddy_in_dom[u]][:,2] - z ) < domain.sigmas[eddy_in_dom[u]][:,k,2] ) )

        if len(eddy_on_line['u'][0]) == 0:
            print('No online for u')
        if len(eddy_on_line['v'][0]) == 0:
            print('No online for v')
        if len(eddy_on_line['w'][0]) == 0:
            print('No online for w')

        primes_no_norm = np.zeros((xs.shape[0],3))

        #Loop over each u,v,w
        for k,u in zip(range(3),['u','v','w']):
            #Travel down line at this location
            for j,x in enumerate(xs):

                #########################
                #Compute u'
                #########################
                #Find all non zero eddies for u,v,w as current time "x"
                x_dist = np.abs( domain.eddy_locs[eddy_in_dom[u]][eddy_on_line[u]][:,0] - x )
                eddy_on_point = np.where( x_dist < domain.sigmas[eddy_in_dom[u]][eddy_on_line[u]][:,k,0] )
                if len(eddy_on_point[0]) == 0:
                    primes_no_norm[j,k] = 0.0
                else:
                    #Compute distances to all contributing points
                    x_dist = np.abs( domain.eddy_locs[eddy_in_dom[u]][eddy_on_line[u]][eddy_on_point][:,0] - x )
                    y_dist = np.abs( domain.eddy_locs[eddy_in_dom[u]][eddy_on_line[u]][eddy_on_point][:,1] - y )
                    z_dist = np.abs( domain.eddy_locs[eddy_in_dom[u]][eddy_on_line[u]][eddy_on_point][:,2] - z )

                    #Collect sigmas from all contributing points
                    x_sigma = domain.sigmas[eddy_in_dom[u]][eddy_on_line[u]][eddy_on_point][:,k,0]
                    y_sigma = domain.sigmas[eddy_in_dom[u]][eddy_on_line[u]][eddy_on_point][:,k,1]
                    z_sigma = domain.sigmas[eddy_in_dom[u]][eddy_on_line[u]][eddy_on_point][:,k,2]

                    #Individual f(x) 'tent' functions for each contributing point
                    fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
                    fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
                    fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

                    #Total f(x) from each contributing point
                    fx  = np.vstack(fxx*fxy*fxz)
                    if normalization == 'jarrin':
                        fx = np.sqrt(domain.VB)/np.sqrt(np.product((x_sigma,y_sigma,z_sigma),axis=0))*fx
                        #Total f(x)
                        fx  = np.sqrt(domain.VB)/np.sqrt(np.product((x_sigma,y_sigma,z_sigma),axis=0)) * fxx*fxy*fxz
                        #Contributing eddy intensities
                        e = domain.eps[eddy_on_line][eddy_on_point]
                        ck = np.matmul(aij, e.T).T
                        #All individual contributions to signal at current point
                        Xk = ck*fx.reshape(fx.shape[0],1)

                        #Fluctuation values at point
                        prime = 1.0/np.sqrt(domain.neddy) * np.sum( Xk ,axis=0)

                        prime_no_norm[j,k] = prime[k]

                    elif normalization == 'exact':
                        ej = domain.eps[eddy_in_dom[u]][eddy_on_line[u]][eddy_on_point]
                        #multiply each eddys function/component by its sign
                        primes_no_norm[j,k] = np.sum( ej*fx ,axis=0)[k] #Only take kth component

        ########################################
        #We now have data over the whole line
        ########################################

        if normalization == 'exact':
            #Contition the total time signals
            #Zero out mean
            primes_no_norm = primes_no_norm - np.mean(primes_no_norm,axis=0)

            #whiten data to eliminate random covariance
            cov = np.cov(primes_no_norm,rowvar=False,bias=True)
            eig_vec = np.linalg.eig(cov)[1]
            primes_no_norm = np.matmul(eig_vec.T,primes_no_norm.T).T

            #Set varianve of each signal to 1
            norm_factor = np.sqrt(np.mean(primes_no_norm**2,axis=0))
            primes_normed = primes_no_norm/norm_factor

            #Force statistics of three signals t match Rij
            prime = np.matmul(L, primes_normed.T).T

        elif normalization == 'jarrin':
            prime = primes_no_norm

        #Return fluctionats
        primes[i] = prime

    return primes
