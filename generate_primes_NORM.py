import numpy as np
import sys
np.random.seed(1010)

def generate_primes(ys,zs,domain,nframes):

    #check if we have eddys or not
    if not hasattr(domain,'eddy_locs'):
        raise ValueError('Please populate your domain before trying to generate fluctiations')

    #Define "time" points for frames
    xs = np.linspace(0,domain.xmax,nframes)
    #Storage for fluctuations
    primes = np.empty((len(ys),nframes,3))

    #Loop over each location
    for i,(y,z) in enumerate(zip(ys,zs)):
        print('   y=',y)
        #Find eddies that contribute on line for y,z
        eddy_on_line_u = np.where( (np.abs( domain.eddy_locs[:,1] - y ) < domain.sigmas[:,0,1] )
                                 & (np.abs( domain.eddy_locs[:,2] - z ) < domain.sigmas[:,0,2] ) )
        eddy_on_line_v = np.where( (np.abs( domain.eddy_locs[:,1] - y ) < domain.sigmas[:,1,1] )
                                 & (np.abs( domain.eddy_locs[:,2] - z ) < domain.sigmas[:,1,2] ) )
        eddy_on_line_w = np.where( (np.abs( domain.eddy_locs[:,1] - y ) < domain.sigmas[:,2,1] )
                                 & (np.abs( domain.eddy_locs[:,2] - z ) < domain.sigmas[:,2,2] ) )

        if len(eddy_on_line_u[0]) == 0:
            print('No online for u')
        if len(eddy_on_line_v[0]) == 0:
            print('No online for v')
        if len(eddy_on_line_w[0]) == 0:
            print('No online for w')

        primes_no_norm = np.zeros((xs.shape[0],3))
        for j,x in enumerate(xs):

            #########################
            #Compute u'
            #########################
            #Find all non zero eddies for u,v,w as current time "x"
            x_dist_u = np.abs( domain.eddy_locs[eddy_on_line_u][:,0] - x )
            eddy_on_point_u = np.where( x_dist_u < domain.sigmas[eddy_on_line_u][:,0,0] )
            if len(eddy_on_point_u[0]) == 0:
                primes_no_norm[j,0] = 0.0
                print(f'No u point on {x}')
            else:
                #Compute distances to all contributing points
                x_dist = np.abs( domain.eddy_locs[eddy_on_line_u][eddy_on_point_u][:,0] - x )
                y_dist = np.abs( domain.eddy_locs[eddy_on_line_u][eddy_on_point_u][:,1] - y )
                z_dist = np.abs( domain.eddy_locs[eddy_on_line_u][eddy_on_point_u][:,2] - z )

                #Collect sigmas from all contributing points
                x_sigma = domain.sigmas[eddy_on_line_u][eddy_on_point_u][:,0,0]
                y_sigma = domain.sigmas[eddy_on_line_u][eddy_on_point_u][:,0,1]
                z_sigma = domain.sigmas[eddy_on_line_u][eddy_on_point_u][:,0,2]

                #Individual f(x) 'tent' functions for each contributing point
                fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
                fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
                fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

                #Total f(x) from each contributing point
                fx  = np.vstack(fxx*fxy*fxz)
                ej = domain.eps[eddy_on_line_u][eddy_on_point_u]
                #multiply each eddys function/component by its sign
                primes_no_norm[j,0] = np.sum( ej*fx ,axis=0)[0] #Only take u component

            #########################
            #Compute v'
            #########################
            x_dist_v = np.abs( domain.eddy_locs[eddy_on_line_v][:,0] - x )
            eddy_on_point_v = np.where( x_dist_v < domain.sigmas[eddy_on_line_v][:,1,0] )
            if len(eddy_on_point_v[0]) == 0:
                primes_no_norm[j,1] = 0.0
                print(f'No v point on {x}')
            else:
                x_dist = np.abs( domain.eddy_locs[eddy_on_line_v][eddy_on_point_v][:,0] - x )
                y_dist = np.abs( domain.eddy_locs[eddy_on_line_v][eddy_on_point_v][:,1] - y )
                z_dist = np.abs( domain.eddy_locs[eddy_on_line_v][eddy_on_point_v][:,2] - z )

                x_sigma = domain.sigmas[eddy_on_line_v][eddy_on_point_v][:,1,0]
                y_sigma = domain.sigmas[eddy_on_line_v][eddy_on_point_v][:,1,1]
                z_sigma = domain.sigmas[eddy_on_line_v][eddy_on_point_v][:,1,2]

                #Individual f(x) 'tent' functions for each contributing point
                fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
                fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
                fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

                #Total f(x) from each contributing point
                fx  = np.vstack(fxx*fxy*fxz)
                ej = domain.eps[eddy_on_line_v][eddy_on_point_v]
                #multiply each eddys function/component by its sign
                primes_no_norm[j,1] = np.sum( ej*fx ,axis=0)[1] #Only take v component

            #########################
            #Compute w'
            #########################
            x_dist_w = np.abs( domain.eddy_locs[eddy_on_line_w][:,0] - x )
            eddy_on_point_w = np.where( x_dist_w < domain.sigmas[eddy_on_line_w][:,2,0] )
            if len(eddy_on_point_w[0]) == 0:
                primes_no_norm[j,2] = 0.0
                print(f'No w point on {x}')
            else:
                x_dist = np.abs( domain.eddy_locs[eddy_on_line_w][eddy_on_point_w][:,0] - x )
                y_dist = np.abs( domain.eddy_locs[eddy_on_line_w][eddy_on_point_w][:,1] - y )
                z_dist = np.abs( domain.eddy_locs[eddy_on_line_w][eddy_on_point_w][:,2] - z )

                x_sigma = domain.sigmas[eddy_on_line_w][eddy_on_point_w][:,2,0]
                y_sigma = domain.sigmas[eddy_on_line_w][eddy_on_point_w][:,2,1]
                z_sigma = domain.sigmas[eddy_on_line_w][eddy_on_point_w][:,2,2]

                #Individual f(x) 'tent' functions for each contributing point
                fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
                fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
                fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

                #Total f(x) from each contributing point
                fx  = np.vstack(fxx*fxy*fxz)
                ej = domain.eps[eddy_on_line_w][eddy_on_point_w]
                #multiply each eddys function/component by its sign
                primes_no_norm[j,2] = np.sum( ej*fx ,axis=0)[2]


        ########################################
        #We now have data over the whole line
        ########################################

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

        #Compute Rij for y location
        Rij = domain.Rij_interp(y)
        #Cholesky decomp of stats
        L = np.linalg.cholesky(Rij)

        #Force statistics of three signals t match Rij
        prime = np.matmul(L, primes_normed.T).T

        #Return fluctionats
        primes[i] = prime

    return primes
