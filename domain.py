import numpy as np

class domain():

    def __init__(self,Ublk,tme,delta,utau,viscosity):

        self.x_length = Ublk*tme
        self.Ublk = Ublk
        self.delta = delta
        self.utau = utau
        self.viscosity = viscosity
        self.yp1 = viscosity/utau

        self.flow_type = None

        self.y_height = None
        self.z_height = None

        self.sigma_interp = None
        self.sigma_x_min = None
        self.sigma_x_max = None
        self.sigma_y_min = None
        self.sigma_y_max = None
        self.sigma_z_min = None
        self.sigma_z_max = None
        self.V_sigma_min = None
        self.V_sigma_max = None

        self.Rij_interp = None
        self.Ubar_interp = None

        self.VB = None
        self.neddy = None
        self.eddy_locs = None
        self.eps = None
        self.sigmas = None

        self.sigmas_from = None
        self.stats_from = None
        self.profile_from = None
        self.eddy_pop_method = None

    def set_sem_data(self,sigmas_from='jarrin',stats_from='moser',profile_from='channel',scale_factor=1.0):

        self.sigmas_from = sigmas_from
        self.stats_from = stats_from
        self.profile_from = profile_from

        #SIGMAS
        if sigmas_from == 'jarrin':
            from .sigmas.jarrin_channel import add_sigmas
        elif sigmas_from == 'uniform':
            from .sigmas.uniform import add_sigmas
        elif sigmas_from == 'linear_bl':
            from .sigmas.linear_bl import add_sigmas
        else:
            raise NameError(f'Unknown sigmas keyword : {sigmas_from}')

        #STATS
        if stats_from == 'moser':
            from .stats.moser_channel import add_stats
        elif stats_from == 'spalart':
            from .stats.spalart_bl import add_stats
        else:
            raise NameError(f'Unknown stats keyword : {stats_from}')

        #MEAN VELOCITY PROFILE
        if profile_from == 'channel':
            from .profiles.channel import add_profile
        elif profile_from == 'bl':
            from .profiles.bl import add_profile
        elif profile_from == 'spalart':
            from .profiles.spalart_bl import add_profile
        else:
            raise NameError(f'Unknown profile keyword : {profile_from}')

        add_sigmas(self, scale_factor)
        add_stats(self)
        add_profile(self)

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
        if hasattr(self,'sigmas'):
            self.sigmas = self.sigma_interp(self.eddy_locs[:,1])

    def print_info(self):
        print(f'Flow Type: {self.flow_type}')
        print('Flow Parameters:')
        print(f'    U_bulk = {self.Ublk} [m/s]')
        print(f'    delta = {self.delta} [m]')
        print(f'    u_tau = {self.utau} [m/s]')
        print(f'    viscosity(nu) = {self.viscosity} [m^2/s]')
        print(f'    Y+ one = {self.yp1} [m]')

        print(f'Sigmas from {self.sigmas_from}')
        print(f'Stats from {self.stats_from}')
        print(f'Profile from {self.profile_from}')

        print('Sigmas (min,max):')
        print(f'    x : {self.sigma_x_min},{self.sigma_x_max}')
        print(f'    y : {self.sigma_y_min},{self.sigma_y_max}')
        print(f'    z : {self.sigma_z_min},{self.sigma_z_max}')
        print(f'  Vol : {self.V_sigma_min},{self.V_sigma_max}')

        print(f'Eddy population method : {self.eddy_pop_method}')
        print(f'Number of Eddys: {self.neddy}')
