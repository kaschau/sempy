import numpy as np
from .misc import progress_bar

def generate_primes(ys,zs,domain,nframes,normalization,convect='uniform',progress=True):
    '''
    Generate Primes Function

    Routine to march down the length (time-line) of mega-box at specified (y,z) coordinate points, stopping along the way
    to calculate the fluctuations at each time point. To improve performace, we first filter out all the eddy in domain
    that cannot possibly contribute to the group of (y,z) points provided based on the eddy's sigmas in y and z.

    Further, for each (y,z) pair we want to compute a signal for, we filter out the eddys that cannot possible contribute
    to the current (y,z) point's time line, again based on the eddy's sigmas in y and z.

    Finally, we can filter out the eddys that do not contribute to an individual point as we march down the time line
    based on the reduced set of eddy's sigmas in x.

    With this very reduced set of eddys that contribute to a point, we compute the fluctuation for that point.


    Parameters:
    -----------
      ys     : numpy.array
            Array of y coordinates of any shape
      zs     : numpy.array
            Array of shape(y) corresponsing one-to-one matching (y,z) pairs
      domain : sempy.domain
            Domain object to fully populated with parameters and dat.
      nframes : int
            Number of frames to generate per (y,z) pair

    Returns:
    --------
        up,vp,wp : numpy.arrrays
            Fluctuation result arrays of shape ( (nframes, shape(y)) ). First axis is 'time', second axis is the
            shape of the input ys and zs arrays.
            So up[0] is the corresponding u fluctiations at all the y points for the first frame.
    '''

    #check if we have eddys or not
    if domain.eddy_locs is None:
        raise ValueError('Please populate your domain before trying to generate fluctiations')

    #Check that nframes is large enough with exact normalization
    if 3 < nframes < 10 and normalization == 'exact':
        print('WARNING: You are using exact normalization with very few framses. Weird things may happen. Consider using jarrin normalization or increasing number of framses.')
    elif nframes <= 3 and normalization == 'exact':
        raise ValueError('Need more frames to use exact normaliation, consider using jarrin or creating more frames.')

    #Check ys and zs are the same shape
    if ys.shape != zs.shape:
        raise TypeError('ys and zs need to be the same shape.')

    #make sure ys and zs are numpy arrays
    ys = np.array(ys)
    zs = np.array(zs)

    #As a sanity check, make sure none of our points are outside the patch
    if (ys.min() < -0.0001*domain.y_height or ys.max() > 1.0001*domain.y_height or
        zs.min() < -0.0001*domain.z_width  or zs.max() > 1.0001*domain.z_width):
        raise ValueError('Woah there, some of your points you are trying to calculate fluctuations for are completely outside your domain!')

    #store the input array shape and then flatten the yz pairs
    yshape = ys.shape
    ys = ys.ravel()
    zs = zs.ravel()

    #We want to filter out all eddys that are not possibly going to contribute to this set of ys and zs
    #we are going to refer to the boundinf box that encapsulates ALL y,z pairs of the current 'patch'
    patch_ymin = ys.min()
    patch_ymax = ys.max()
    patch_zmin = zs.min()
    patch_zmax = zs.max()
    eddy_locs_in_patch = dict()
    sigmas_in_patch = dict()
    eps_in_patch = dict()

    # Because u,v,w fluctuations each have different values of sigma_x,y,z we need to keep seperate lists of the
    # eddys that can possible contribute to the patch. We also need to track the sigmas and epsilons that correspond
    # to these eddy locations.
    for k,u in zip(range(3),['u','v','w']):
        eddys_in_patch = np.where( ( domain.eddy_locs[:,1] - domain.sigmas[:,k,1] < patch_ymax )
                                 & ( domain.eddy_locs[:,1] + domain.sigmas[:,k,1] > patch_ymin )
                                 & ( domain.eddy_locs[:,2] - domain.sigmas[:,k,2] < patch_zmax )
                                 & ( domain.eddy_locs[:,2] + domain.sigmas[:,k,2] > patch_zmin ) )
        eddy_locs_in_patch[u] = domain.eddy_locs[eddys_in_patch]
        sigmas_in_patch[u] = domain.sigmas[eddys_in_patch]
        eps_in_patch[u] = domain.eps[eddys_in_patch]

    ######################################################################
    # We now have a reduced set of eddys that overlap the current patc
    ######################################################################
    if progress:
        print(f'Searching a reduced set of {eddy_locs_in_patch["u"].shape[0]} u eddies, {eddy_locs_in_patch["v"].shape[0]} v eddies, and {eddy_locs_in_patch["w"].shape[0]} w eddies using {convect} convection speed')
    #Storage for fluctuations
    up = np.empty((nframes,len(ys)))
    vp = np.empty((nframes,len(ys)))
    wp = np.empty((nframes,len(ys)))

    #just counter for progress display
    total = len(ys)
    #Define "time" points for frames, if its uniform, we only need to do this once.
    if convect == 'uniform':
        xs = np.linspace(0,domain.x_length,nframes)
    #Loop over each location
    for i,(y,z) in enumerate(zip(ys,zs)):

        zero_online = {'u':False,'v':False,'w':False}
        #Compute Rij for current y location
        Rij = domain.Rij_interp(y)
        #Cholesky decomp of stats
        L = np.linalg.cholesky(Rij)
        if progress:
            progress_bar(i+1,total,'Generating Primes')
        #Find eddies that contribute on the current y,z line. This search is done on the reduced set of
        # eddys filtered on the "patch"
        eddy_locs_on_line = dict()
        sigmas_on_line = dict()
        eps_on_line = dict()
        for k,u in zip(range(3),['u','v','w']):
            eddys_on_line = np.where( (np.abs( eddy_locs_in_patch[u][:,1] - y ) < sigmas_in_patch[u][:,k,1] )
                                    & (np.abs( eddy_locs_in_patch[u][:,2] - z ) < sigmas_in_patch[u][:,k,2] ) )
            eddy_locs_on_line[u] = eddy_locs_in_patch[u][eddys_on_line]
            sigmas_on_line[u] = sigmas_in_patch[u][eddys_on_line]
            eps_on_line[u] = eps_in_patch[u][eddys_on_line]

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
        #Define "time" points for frames, if its local, we need to recalculate for each y location.
        if convect == 'local':
            local_Ubar = domain.Ubar_interp(y)
            length = domain.x_length*local_Ubar/domain.Ublk
            xs = np.linspace(0,length,nframes)
        #Storage for un-normalized fluctuations
        primes_no_norm = np.zeros((xs.shape[0],3))

        empty_pts = 0 #counter for empty points
        #Loop over each u,v,w
        for k,u in zip(range(3),['u','v','w']):
            #If this line has no eddys on it, move on
            if zero_online[u]:
                continue

            if convect == 'local':
               #We need each eddys individual Ubar for the offset calculated below
               local_eddy_Ubar = domain.Ubar_interp(eddy_locs_on_line[u][:,1])

            #Travel down line at this y,z location
            for j,x in enumerate(xs):
                zero_onpoint = {'u':False,'v':False,'w':False}

                if convect == 'local':
                    #This may be tough to explain, but just draw it out for yourself and you'll
                    #figure it out:

                    #If we want each eddy to convect with its own local velocity
                    #instead of the Ublk velocity, it is not enough to just traverse through the
                    #mega box at the profile Ubar for the current location's y height. This is
                    #because the eddys located slightly above/below the location we are
                    #traversing down (that contribute to fluctuations) are
                    #moving at different speeds than our current point of interest. So we need to
                    #calculate an offset to account for that fact that as we have traversed through the
                    #domain at the local convective speed, the faster eddys will approach
                    #us slightly more quickly, while the slower eddys
                    #will approach us more slowly. This x offset will account for that and is merely the difference
                    #in convection speeds between the current line we are traversing down, and the speeds of
                    #the individual eddys.
                    x_offset = (local_Ubar - local_eddy_Ubar)/local_Ubar * x
                else:
                    #If all the eddys convect at the same speed, then traversing through the mega box at Ublk
                    #is identical so the eddys convecting by us at Ublk, so there is no need to offset any
                    #eddy positions
                    x_offset = np.zeros(eddy_locs_on_line[u].shape[0])

                #########################
                #Compute u'
                #########################
                #Find all non zero eddies for u,v,w at current time "x"
                x_dist = np.abs( (eddy_locs_on_line[u][:,0]+x_offset) - x )
                eddys_on_point = np.where( x_dist < sigmas_on_line[u][:,k,0] )
                x_offset = x_offset[eddys_on_point]
                if len(eddys_on_point[0]) == 0:
                    empty_pts += 1
                    primes_no_norm[j,k] = 0.0
                else:

                    ######################################################################
                    # We now have a reduced set of eddys that overlap the current point
                    ######################################################################
                    #Compute distances to all contributing points
                    x_dist = np.abs( ( eddy_locs_on_line[u][eddys_on_point][:,0]+x_offset) - x )
                    y_dist = np.abs(   eddy_locs_on_line[u][eddys_on_point][:,1]           - y )
                    z_dist = np.abs(   eddy_locs_on_line[u][eddys_on_point][:,2]           - z )

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

        if empty_pts > 10:
            print(f'Warning, {empty_pts} points with zero fluctuations detected at y={y},z={z}\n')

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

            #Set variance of each signal to 1
            norm_factor = np.sqrt(np.mean(primes_no_norm**2,axis=0))
            primes_normed = primes_no_norm/norm_factor

        elif normalization == 'jarrin':
            primes_normed = np.sqrt(domain.VB)/np.sqrt(domain.neddy) * primes_no_norm
        elif normalization == 'none':
            primes_normed = primes_no_norm
        else:
            raise NameError(f'Error: Unknown normalization : {normalization}')

        #Multiply normalized signal by stats
        prime = np.matmul(L, primes_normed.T).T

        #Return fluctionats
        up[:,i] = prime[:,0]
        vp[:,i] = prime[:,1]
        wp[:,i] = prime[:,2]


    up.reshape(tuple([nframes]+list(yshape)))
    vp.reshape(tuple([nframes]+list(yshape)))
    wp.reshape(tuple([nframes]+list(yshape)))
    return up,vp,wp
