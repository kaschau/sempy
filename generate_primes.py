import numpy as np
from scipy import interpolate as itrp
from .misc import progress_bar
from .norm import *

def generate_primes(ys,zs,domain,nframes,normalization,interpolate=False,convect='uniform',progress=True):
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
            Domain object fully populated with parameters and data.
      nframes : int
            Number of frames to generate per (y,z) pair
      normalization : str
            String corresponding to the method of normalizing the signal.
            Available options are:
               'jarrin' : Approximates an integral with Vbox/(neddy*V_eddy)
               'exact'  : Produce exact statistics by bending the signal to your will
               'none'   : Return the raw sum of the eddys for each point
      interpolation : bool
            T/F flag to determine whether your frame rate is high enough to approximate the
            continuous signal, thus no significalt loss of statistics, or if you plan to interpolate
            between frames. If True, a time signal is "pre-interpolated" and statistics are imposed
            on that pre-interpolated signal, then just the frame values are returned. If False, nothing
            is done
      convect : str
            String corresponding to the method of convection through the flow.
            Current options are:
               'uniform' : Convect through mega-volume at Ublk everywhere
               'local'   : Convect through mega-volume at local convective speed based on domain.Ubar_interp(y)
      progress : bool
            Whether to display progress bar

    Returns:
    --------
        up,vp,wp : numpy.arrrays
            Fluctuation result arrays of shape ( (nframes, shape(y)) ). First axis is 'time', second axis is the
            shape of the input ys and zs arrays.
            So up[0] is the corresponding u fluctiations at all the y,z points for the first frame.
    '''

    #check if we have eddys or not
    if domain.eddy_locs is None:
        raise RuntimeError('Please populate your domain before trying to generate fluctiations')

    #Check that nframes is large enough with exact normalization
    if 3 < nframes < 10 and normalization == 'exact':
        print('WARNING: You are using exact normalization with very few framses. Weird things may happen. Consider using jarrin normalization or increasing number of framses.')
    elif nframes <= 3 and normalization == 'exact':
        raise RuntimeError('Need more frames to use exact normaliation, consider using jarrin or creating more frames.')

    #Check ys and zs are the same shape
    if ys.shape != zs.shape:
        raise RuntimeError('ys and zs need to be the same shape.')

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
    #we are going to refer to the bounding box that encapsulates ALL y,z pairs of the current 'patch'
    patch_ymin = ys.min()
    patch_ymax = ys.max()
    patch_zmin = zs.min()
    patch_zmax = zs.max()

    eddys_in_patch = np.where( ( domain.eddy_locs[:,1] - np.max(domain.sigmas[:,:,1],axis=1) < patch_ymax )
                             & ( domain.eddy_locs[:,1] + np.max(domain.sigmas[:,:,1],axis=1) > patch_ymin )
                             & ( domain.eddy_locs[:,2] - np.max(domain.sigmas[:,:,2],axis=1) < patch_zmax )
                             & ( domain.eddy_locs[:,2] + np.max(domain.sigmas[:,:,2],axis=1) > patch_zmin ) )
    eddy_locs_in_patch = domain.eddy_locs[eddys_in_patch]
    sigmas_in_patch = domain.sigmas[eddys_in_patch]
    eps_in_patch = domain.eps[eddys_in_patch]

    ######################################################################
    # We now have a reduced set of eddys that overlap the current patch
    ######################################################################
    # if progress:
    #     print(f'Searching a reduced set of {eddy_locs_in_patch["u"].shape[0]} u eddies, {eddy_locs_in_patch["v"].shape[0]} v eddies, and {eddy_locs_in_patch["w"].shape[0]} w eddies using {convect} convection speed')
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

        zero_online = False

        if progress:
            progress_bar(i+1,total,'Generating Primes')
        #Find eddies that contribute on the current y,z line. This search is done on the reduced set of
        # eddys filtered on the "patch"
        eddys_on_line = np.where( (np.abs( eddy_locs_in_patch[:,1] - y ) < sigmas_in_patch[:,k,1] )
                                & (np.abs( eddy_locs_in_patch[:,2] - z ) < sigmas_in_patch[:,k,2] ) )
        eddy_locs_on_line = eddy_locs_in_patch[eddys_on_line]
        sigmas_on_line = sigmas_in_patch[eddys_on_line]
        eps_on_line = eps_in_patch[eddys_on_line]

        #We want to know if an entire line has zero eddys, this will be annoying for BL in the free stream
        # so we will only print out a warning for the BL cases if the value of y is below the BL thickness
        if len(eddy_locs_on_line) == 0:
            zero_online = True
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

        #If this line has no eddys on it, move on
        if zero_online:
            continue

        if convect == 'local':
           #We need each eddys individual Ubar for the offset calculated below
           local_eddy_Ubar = domain.Ubar_interp(eddy_locs_on_line[:,1])

        #Travel down line at this y,z location
        for j,x in enumerate(xs):
            zero_onpoint = False

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
                x_offset = np.zeros(eddy_locs_on_line.shape[0])

            #########################
            #Compute u'
            #########################
            #Find all non zero eddies for u,v,w at current time "x"
            x_dist = np.abs( (eddy_locs_on_line[:,0]+x_offset) - x )
            eddys_on_point = np.where( x_dist < np.max(sigmas_on_line[:,:,0],axis=1) )
            x_offset = x_offset[eddys_on_point]
            if len(eddys_on_point[0]) == 0:
                empty_pts += 1
                primes_no_norm[j,:] = 0.0
                continue

            ######################################################################
            # We now have a reduced set of eddys that overlap the current point
            ######################################################################
            #Compute distances to all contributing points
            x_dist = np.abs( ( eddy_locs_on_line[eddys_on_point][:,0]+x_offset) - x )
            y_dist = np.abs(   eddy_locs_on_line[eddys_on_point][:,1]           - y )
            z_dist = np.abs(   eddy_locs_on_line[eddys_on_point][:,2]           - z )

            #Collect sigmas from all contributing points
            x_sigma = sigmas_on_line[eddys_on_point][:,:,0]
            y_sigma = sigmas_on_line[eddys_on_point][:,:,1]
            z_sigma = sigmas_on_line[eddys_on_point][:,:,2]

            #Individual f(x) 'tent' functions for each contributing point
            fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
            fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
            fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

            #Total f(x) from each contributing point
            fx  = fxx*fxy*fxz
            if normalization == 'jarrin':
                fx = 1.0/np.sqrt(np.product((x_sigma,y_sigma,z_sigma),axis=0)) * fx

            fx = fx.reshape((fx.shape[0],1))
            ej = eps_on_line[u][eddys_on_point]
            #multiply each eddys function/component by its sign
            primes_no_norm[j,k] = np.sum( ej*fx ,axis=0)[k] #Only take kth component

        if empty_pts > 10:
            print(f'Warning, {empty_pts} points with zero fluctuations detected at y={y},z={z}\n')

        ########################################
        #We now have data over the whole line
        ########################################

        #If we are planning on interpolating the signal for a raptor simulaiton, we need to
        # approximate the interpolation here, then normalize the interpolated signal.
        # Otherwise the stats of the interpolated signal in raptor will under represent the
        # desired statistics.
        if interpolate:
            #Current we approximate with 10 points between frames. Could be experimented with
            pts_btw_frames = 10
            temp_N = [j for j in range(nframes)]
            primes_interp = itrp.CubicSpline(temp_N,primes_no_norm,bc_type='not-a-knot',axis=0)
            intrp_N = np.linspace(0,nframes-1,(nframes-1)*(pts_btw_frames+1)+1)
            primes_no_norm = primes_interp(intrp_N)
            frame_indicies = tuple([j*(pts_btw_frames+1) for j in range(nframes)])
        else:
            frame_indicies = tuple([j for j in range(nframes)])

        #Normalize the total time signals
        if normalization == 'exact':
            primes_normed = exact_norm(primes_no_norm)
        elif normalization == 'jarrin':
            primes_normed = jarrin_norm(primes_no_norm,domain)
        elif normalization == 'none':
            primes_normed = primes_no_norm
        else:
            raise NameError(f'Error: Unknown normalization : {normalization}')

        #Compute Rij for current y location
        Rij = domain.Rij_interp(y)
        #Cholesky decomp of stats
        L = np.linalg.cholesky(Rij)
        #Multiply normalized signal by stats
        prime = np.matmul(L, primes_normed.T).T

        #Keep only the points on the frames
        up[:,i] = prime[frame_indicies,0]
        vp[:,i] = prime[frame_indicies,1]
        wp[:,i] = prime[frame_indicies,2]

    #Return fluctuations
    up = up.reshape(tuple([nframes]+list(yshape)))
    vp = vp.reshape(tuple([nframes]+list(yshape)))
    wp = wp.reshape(tuple([nframes]+list(yshape)))

    return up,vp,wp
