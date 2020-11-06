import numpy as np
import sys
np.random.seed(1010)

def generate_primes(ys,zs,domain,nframes,normalization='exact'):

    #check if we have eddys or not
    if not hasattr(domain,'eddy_locs'):
        raise ValueError('Please populate your domain before trying to generate fluctiations')
    #make sure ys and zs are numpy arrays
    ys = np.array(ys)
    zs = np.array(zs)
    #We want to filter out all eddys that are not possibly going to contribute to this set of ys and zs
    #we are going to refer to the boundinf box that encapsulates ALL y,z pairs ad the 'domain' or 'dom'
    dom_ymin = ys.min()
    dom_ymax = ys.max()
    dom_zmin = zs.min()
    dom_zmax = zs.max()
    eddy_in_dom = dict()
    eddy_locs_in_dom = dict()
    sigmas_in_dom = dict()
    eps_in_dom = dict()

    # Because u,v,w fluctuations each have differenc values of sigma_x,y,z we need to keep seperate lists of the
    # eddys that can possible contribute to the domain. We also need to track the sigmas and epsilons that correspond
    # to these eddy locations.
    for k,u in zip(range(3),['u','v','w']):
        eddys_in_dom = np.where( ( domain.eddy_locs[:,1] < dom_ymax + domain.sigmas[:,k,1].max() )
                              & ( domain.eddy_locs[:,1] > dom_ymin - domain.sigmas[:,k,1].max() )
                              & ( domain.eddy_locs[:,2] < dom_zmax + domain.sigmas[:,k,2].max() )
                              & ( domain.eddy_locs[:,2] > dom_zmin - domain.sigmas[:,k,2].max() ) )
        eddy_locs_in_dom[u] = domain.eddy_locs[eddys_in_dom]
        sigmas_in_dom[u] = domain.sigmas[eddys_in_dom]
        eps_in_dom[u] = domain.eps[eddys_in_dom]

    ######################################################################
    # We now have a reduced set of eddys that overlap the current domain
    ######################################################################

    #Define "time" points for frames
    xs = np.linspace(0,domain.x_length,nframes)
    #Storage for fluctuations
    primes = np.empty((len(ys),nframes,3))

    #Loop over each location
    for i,(y,z) in enumerate(zip(ys,zs)):
        zero_online = {'u':False,'v':False,'w':False}
        #Compute Rij for current y location
        Rij = domain.Rij_interp(y)
        #Cholesky decomp of stats
        L = np.linalg.cholesky(Rij)

        print('   y=',y)

        #Find eddies that contribute on the current y,z line. This search is done on the reduced set of
        # eddys filtered out into the "domain"
        eddy_locs_on_line = dict()
        sigmas_on_line = dict()
        eps_on_line = dict()
        for k,u in zip(range(3),['u','v','w']):
            eddys_on_line = np.where( (np.abs( eddy_locs_in_dom[u][:,1] - y ) < sigmas_in_dom[u][:,k,1] )
                                   & (np.abs( eddy_locs_in_dom[u][:,2] - z ) < sigmas_in_dom[u][:,k,2] ) )
            eddy_locs_on_line[u] = eddy_locs_in_dom[u][eddys_on_line]
            sigmas_on_line[u] = sigmas_in_dom[u][eddys_on_line]
            eps_on_line[u] = eps_in_dom[u][eddys_on_line]

        #We want to know if an entire line has zero eddys, this will be annoying for BL in the free stream
        # so we will only print out a warning for the BL cases if the value of y is below the BL thickness
        for u in ['u','v','w']:
            if len(eddy_locs_on_line[u]) == 0:
                zero_online[u] = True
                if domain.flow_type != 'bl' or y<domain.delta:
                    print(f'Warning, no eddys detected on entire time line at y={y},z={z}')

        ######################################################################
        # We now have a reduced set of eddys that overlap the current y,z line
        ######################################################################

        #Storage for un-normalized fluctuations
        primes_no_norm = np.zeros((xs.shape[0],3))

        #Loop over each u,v,w
        for k,u in zip(range(3),['u','v','w']):
            #If this line has no eddys on it, move on
            if zero_online[u]:
                continue
            #Travel down line at this y,z location
            for j,x in enumerate(xs):
                zero_onpoint = {'u':False,'v':False,'w':False}
                #########################
                #Compute u'
                #########################
                #Find all non zero eddies for u,v,w as current time "x"
                x_dist = np.abs( eddy_locs_on_line[u][:,0] - x )
                eddys_on_point = np.where( x_dist < sigmas_on_line[u][:,k,0] )
                if len(eddys_on_point[0]) == 0:
                    print(f'Warning, no eddys detected on time point at x={x},y={y},z={z}')
                    primes_no_norm[j,k] = 0.0
                else:

                    ######################################################################
                    # We now have a reduced set of eddys that overlap the current point
                    ######################################################################
                    #Compute distances to all contributing points
                    x_dist = np.abs( eddy_locs_on_line[u][eddys_on_point][:,0] - x )
                    y_dist = np.abs( eddy_locs_on_line[u][eddys_on_point][:,1] - y )
                    z_dist = np.abs( eddy_locs_on_line[u][eddys_on_point][:,2] - z )

                    #Collect sigmas from all contributing points
                    x_sigma = sigmas_on_line[u][eddys_on_point][:,k,0]
                    y_sigma = sigmas_on_line[u][eddys_on_point][:,k,1]
                    z_sigma = sigmas_on_line[u][eddys_on_point][:,k,2]

                    #Individual f(x) 'tent' functions for each contributing point
                    fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
                    fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
                    fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

                    #Total f(x) from each contributing point
                    fx  = fxx*fxy*fxz
                    if normalization == 'jarrin':
                        fx = 1.0/np.sqrt(np.product((x_sigma,y_sigma,z_sigma),axis=0)) * fx

                    fx = np.vstack(fx)
                    ej = eps_on_line[u][eddys_on_point]
                    #multiply each eddys function/component by its sign
                    primes_no_norm[j,k] = np.sum( ej*fx ,axis=0)[k] #Only take kth component

        ########################################
        #We now have data over the whole line
        ########################################

        if normalization == 'exact':
            #Condition the total time signals
            #Zero out mean
            primes_no_norm = primes_no_norm - np.mean(primes_no_norm,axis=0)

            #whiten data to eliminate random covariance
            cov = np.cov(primes_no_norm,rowvar=False,bias=True)
            eig_vec = np.linalg.eig(cov)[1]
            primes_no_norm = np.matmul(eig_vec.T,primes_no_norm.T).T

            #Set varianve of each signal to 1
            norm_factor = np.sqrt(np.mean(primes_no_norm**2,axis=0))
            primes_normed = primes_no_norm/norm_factor

        elif normalization == 'jarrin':
            primes_normed = np.sqrt(domain.VB)/np.sqrt(domain.neddy) * primes_no_norm

        #Force statistics of three signals t match Rij
        prime = np.matmul(L, primes_normed.T).T
        #Return fluctionats
        primes[i] = prime

    return primes
