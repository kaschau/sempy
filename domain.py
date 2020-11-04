import numpy as np

class domain():

    def __init__(self,Ublk,tme,delta,utau,viscosity):

        self.x_length = Ublk*tme
        self.Ublk = Ublk
        self.delta = delta
        self.utau = utau
        self.viscosity = viscosity
        self.yp1 = viscosity/utau

    def set_sem_data(self,sigmas_from='jarrin',stats_from='moser',scale_factor=1.0):

        if sigmas_from == 'jarrin':
            from sigmas.jarrin_channel import add_sigma_info
        elif sigmas_from == 'uniform':
            from sigmas.uniform import add_sigma_info
        elif sigmas_from == 'linear_bl':
            from sigmas.linear_bl import add_sigma_info

        if stats_from == 'moser':
            from stats.moser_channel import add_stat_info
        if stats_from == 'spalart':
            from stats.spalart_bl import add_stat_info

        add_sigma_info(self, scale_factor)
        add_stat_info(self)


    def compute_sigmas(self):
        #Compute all eddy sigmas as function of y
        self.sigmas = self.sigma_interp(self.eddy_locs[:,1])
    def generate_eps(self):
        #generate epsilons
        self.eps = np.where(np.random.uniform(low=-1,high=1,size=(self.neddy,3))<= 0.0, -1.0,1.0)

    def make_periodic(self, periodic_x=False, periodic_y=False, periodic_z=False):

        #check if we have epsilons or not
        if not hasattr(self,'eps'):
            raise AttributeError('Please generate your domains epsilons before making it periodic')
        #Make periodic
        if periodic_x:
            keep_eddys = np.where(self.eddy_locs[:,0] < self.x_length - self.sigma_x_max)
            keep_eddy_locs = self.eddy_locs[keep_eddys]
            keep_eps = self.eps[keep_eddys]
            periodic_eddys = np.where(self.eddy_locs[:,0] < self.sigma_x_max)
            periodic_eddy_locs = self.eddy_locs[periodic_eddys]
            periodic_eddy_locs[:,0] = periodic_eddy_locs[:,0] + self.x_length
            periodic_eps = self.eps[periodic_eddys]
            self.eddy_locs = np.concatenate( (keep_eddy_locs, periodic_eddy_locs) )
            self.eps = np.concatenate( (keep_eps, periodic_eps) )
        if periodic_y:
            keep_eddys = np.where(self.eddy_locs[:,1] < self.y_height - self.sigma_y_max)
            keep_eddy_locs = self.eddy_locs[keep_eddys]
            keep_eps = self.eps[keep_eddys]
            periodic_eddys = np.where(self.eddy_locs[:,1] < self.sigma_y_max)
            periodic_eddy_locs = self.eddy_locs[periodic_eddys]
            periodic_eddy_locs[:,1] = periodic_eddy_locs[:,1] + self.y_height
            periodic_eps = self.eps[periodic_eddys]
            self.eddy_locs = np.concatenate( (keep_eddy_locs, periodic_eddy_locs) )
            self.eps = np.concatenate( (keep_eps, periodic_eps) )
        if periodic_z:
            keep_eddys = np.where(self.eddy_locs[:,2] < self.z_width - self.sigma_z_max)
            keep_eddy_locs = self.eddy_locs[keep_eddys]
            keep_eps = self.eps[keep_eddys]
            periodic_eddys = np.where(self.eddy_locs[:,2] < self.sigma_z_max)
            periodic_eddy_locs = self.eddy_locs[periodic_eddys]
            periodic_eddy_locs[:,2] = periodic_eddy_locs[:,2] + self.z_width
            periodic_eps = self.eps[periodic_eddys]
            self.eddy_locs = np.concatenate( (keep_eddy_locs, periodic_eddy_locs) )
            self.eps = np.concatenate( (keep_eps, periodic_eps) )

        #Update the number of eddys
        self.neddy = self.eddy_locs.shape[0]
        #Update the sigma array if it has been created already
        #check if we have epsilons or not
        if hasattr(self,'sigmas'):
            self.sigmas = self.sigma_interp(self.eddy_locs[:,1])
