import numpy as np
np.random.seed(1010)

class channel:

    def __init__(self,tme,Ublk,ymin,ymax,zmin,zmax,delta,utau):

        self.xmin = 0
        self.xmax = Ublk*tme
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

        self.delta = delta
        self.utau = utau

    def set_flow_data(self,sigmas_from='jarrin',stats_from='moser'):

        if sigmas_from == 'jarrin':
            from sigmas.jarrin_channel import create_sigma_interps
        if stats_from == 'moser':
            from stats.moser_channel import create_stat_interps

        self.sigma_interp = create_sigma_interps(self.delta,self.utau,self.ymin)
        self.Rij_interp,self.Ubar_interp = create_stat_interps(self.delta,self.utau,self.ymin)


    def populate(self, C_Eddy=1.0, periodic_x=False, periodic_y=False, periodic_z=False):

        if not hasattr(self,'sigma_interp'):
            raise ValueError('Please set your flow data before trying to populate your domain')

        #determine min,max sigmas
        test_sigmas = self.sigma_interp(np.linspace(self.ymin,self.ymax,200))

        sigma_x_min = np.min(test_sigmas[:,:,0])
        sigma_x_max = np.max(test_sigmas[:,:,0])
        sigma_y_min = np.min(test_sigmas[:,:,1])
        sigma_y_max = np.max(test_sigmas[:,:,1])
        sigma_z_min = np.min(test_sigmas[:,:,2])
        sigma_z_max = np.max(test_sigmas[:,:,2])

        V_sigma_x_min = np.min(np.product(test_sigmas[:,0,:], axis=1))
        V_sigma_y_min = np.min(np.product(test_sigmas[:,1,:], axis=1))
        V_sigma_z_min = np.min(np.product(test_sigmas[:,2,:], axis=1))

        V_sigma_min = np.min([V_sigma_x_min,V_sigma_y_min,V_sigma_z_min])

        # sigma_x_min = np.min(self.sigma_interp(0)[0])
        # sigma_x_max = np.max(self.sigma_interp(1.0)[0])

        # sigma_y_min = np.min(self.sigma_interp(0)[1])
        # sigma_y_max = np.max(self.sigma_interp(1.0)[1])

        # sigma_z_min = np.min(self.sigma_interp(0)[2])
        # sigma_z_max = np.max(self.sigma_interp(1.0)[2])

        #generate eddy volume
        lows  = [self.xmin - sigma_x_max, self.ymin - sigma_y_max, self.zmin - sigma_z_max]
        highs = [self.xmax + sigma_x_max, self.ymax + sigma_y_max, self.zmax + sigma_z_max]

        self.VB = np.product(np.array(highs) - np.array(lows))

        #Compute number of eddys
        neddy = int( C_Eddy * self.VB / V_sigma_min )
        print(neddy)
        #Generate random eddy locations
        temp_eddy_locs = np.random.uniform(low=lows,high=highs,size=(neddy,3))

        #Make periodic
        if periodic_x:
            temp_eddy_locs = temp_eddy_locs[np.where(temp_eddy_locs[:,0] < self.xmax-sigma_x_max)]
            periodic_eddys = temp_eddy_locs[np.where(temp_eddy_locs[:,0] < self.xmin+sigma_x_max)]
            periodic_eddys[:,0] = periodic_eddys[:,0] + self.xmax
            temp_eddy_locs = np.concatenate( (temp_eddy_locs, periodic_eddys) )
        if periodic_y:
            temp_eddy_locs = temp_eddy_locs[np.where(temp_eddy_locs[:,1] < self.ymax-sigma_y_max)]
            periodic_eddys = temp_eddy_locs[np.where(temp_eddy_locs[:,1] < self.ymin+sigma_y_max)]
            periodic_eddys[:,1] = periodic_eddys[:,1] + self.ymax
            temp_eddy_locs = np.concatenate( (temp_eddy_locs, periodic_eddys) )
        if periodic_z:
            temp_eddy_locs = temp_eddy_locs[np.where(temp_eddy_locs[:,2] < self.zmax-sigma_z_max)]
            periodic_eddys = temp_eddy_locs[np.where(temp_eddy_locs[:,2] < self.zmin+sigma_z_max)]
            periodic_eddys[:,0] = periodic_eddys[:,2] + self.zmax
            temp_eddy_locs = np.concatenate( (temp_eddy_locs, periodic_eddys) )

        self.eddy_locs = temp_eddy_locs
        self.neddy = self.eddy_locs.shape[0]
        #Compute all eddy sigmas as function of y
        self.sigmas = self.sigma_interp(self.eddy_locs[:,1])
        #Compute Rij for all eddys as a function of y
        self.Rij = self.Rij_interp(self.eddy_locs[:,1])
        #Cholesky decomp of all eddys
        self.aij = np.linalg.cholesky(self.Rij)

        #generate epsilons
        temp_eps_k = np.where(np.random.random((self.neddy,3,1)) <= 0.5,1.0,-1.0)
        #Make periodic
        if periodic_x:
            temp_eps_k[np.where(temp_eddy_locs[:,0] > self.xmax-sigma_x_max)] = temp_eps_k[np.where(self.eddy_locs[:,0] < self.xmin+sigma_x_max)]
        if periodic_y:
            temp_eps_k[np.where(temp_eddy_locs[:,1] > self.ymax-sigma_y_max)] = temp_eps_k[np.where(self.eddy_locs[:,1] < self.ymin+sigma_y_max)]
        if periodic_z:
            temp_eps_k[np.where(temp_eddy_locs[:,2] > self.zmax-sigma_z_max)] = temp_eps_k[np.where(self.eddy_locs[:,2] < self.zmin+sigma_z_max)]

        self.eps_k = temp_eps_k

        print(f'Eddy box volume of {self.VB}')
        print(f'Generating {self.neddy} eddys')
        print('Using:')
        print(f'      sigma_x_min = {sigma_x_min}')
        print(f'      sigma_x_max = {sigma_x_max}')
        print(f'      sigma_y_min = {sigma_y_min}')
        print(f'      sigma_y_max = {sigma_y_max}')
        print(f'      sigma_z_min = {sigma_z_min}')
        print(f'      sigma_z_max = {sigma_z_max}')
        print(f'      V_sigma_min = {V_sigma_min}')
