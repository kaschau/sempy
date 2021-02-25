import numpy as np
import scipy.integrate as integrate
from sempy.domain import domain
import sys

class box(domain):

    def __init__(self,flow_type,Ublk,tme,y_height,z_width,delta,utau,viscosity):

        super().__init__(Ublk,tme,delta,utau,viscosity)
        self.y_height = y_height
        self.z_width  = z_width
        self.flow_type = flow_type
        self.randseed = np.random.RandomState(10101)

    def populate(self, C_Eddy=1.0, method='random'):

        if self.sigma_interp is None:
            raise ValueError('Please set your flow data before trying to populate your domain')

        self.eddy_pop_method = method

        #generate eddy volume
        if self.flow_type in ['channel', 'freeshear']:
            lows  = [          0.0 - self.sigma_x_max/2.0,           0.0 - self.sigma_y_max/2.0,          0.0 - self.sigma_z_max/2.0]
            highs = [self.x_length + self.sigma_x_max/2.0, self.y_height + self.sigma_y_max/2.0, self.z_width + self.sigma_z_max/2.0]
        elif self.flow_type == 'bl':
            lows  = [          0.0 - self.sigma_x_max/2.0,           0.0 - self.sigma_y_max/2.0,          0.0 - self.sigma_z_max/2.0]
            highs = [self.x_length + self.sigma_x_max/2.0,    self.delta + self.sigma_y_max/2.0, self.z_width + self.sigma_z_max/2.0]

        if method == 'random':
            #Compute number of eddys
            neddy = int( C_Eddy * self.VB / self.V_sigma_min )
            #Generate random eddy locations
            self.eddy_locs = self.randseed.uniform(low=lows,high=highs,size=(neddy,3))

        elif method == 'PDF':
            #Eddy heights as a function of y
            test_ys = np.linspace(lows[1], highs[1], 200)
            test_sigmas = self.sigma_interp(test_ys)
            #Smallest eddy y height
            Y_eddy = np.min(test_sigmas[:,:,1], axis=1)
            # Find the smallest eddy y and largest eddy y
            Y_eddy_min = Y_eddy.min()
            Y_eddy_max = Y_eddy.max()
            #This ratio sets how much more likely the small eddy placement is compared to the large eddy placement
            Y_ratio = Y_eddy_max/Y_eddy_min
            #Flip and shift so largest eddy sits on zero
            Y_eddy_prime = - ( Y_eddy - Y_eddy_min ) + (Y_eddy_max-Y_eddy_min)

            #Rescale so lowest point is equal to one
            Y_eddy_prime = Y_eddy_prime/Y_eddy_max*Y_ratio + 1.0
            Y_eddy_norm = integrate.trapz(Y_eddy_prime,test_ys)
            #Create a PDF of eddy placement in y by normalizing pdf integral
            pdf = Y_eddy_prime/Y_eddy_norm
            #Compute average eddy volume
            expected_Veddy = integrate.trapz(pdf*Y_eddy,test_ys)**3
            #Compute neddy
            neddy = int( C_Eddy * self.VB / expected_Veddy )
            #Create bins for placing eddys in y
            dy = np.abs(test_ys[1]-test_ys[0])
            bin_centers = 0.5*( test_ys[1::] + test_ys[0:-1] )
            bin_pdf = 0.5*( pdf[1::] + pdf[0:-1] ) * dy
            #place neddys in bins according to pdf
            bin_loc = self.randseed.choice(np.arange(bin_pdf.shape[0]),p=bin_pdf,size=neddy)

            #create y values for eddys
            #can only generate values at bin centers, so we add a random value to randomize y placement within the bin
            eddy_ys = np.take(bin_centers, bin_loc) + self.randseed.uniform(low=-dy/2.0,high=dy/2.0,size=neddy)

            #Generate random eddy locations for their x and z locations
            self.eddy_locs = self.randseed.uniform(low=lows,high=highs,size=(neddy,3))
            #Replace ys with PDF ys
            self.eddy_locs[:,1] = eddy_ys

        else:
            raise NameError(f'Unknown population method : {method}')

        # Now, we remove all eddies whose volume of influence lies completely outside the geometry.
        keep_eddys = np.where( (self.eddy_locs[:,0] + np.max(self.sigma_interp(self.eddy_locs[:,1])[:,:,0],axis=1) > 0.0 ))
        self.eddy_locs = self.eddy_locs[keep_eddys]
        keep_eddys = np.where( (self.eddy_locs[:,0] - np.max(self.sigma_interp(self.eddy_locs[:,1])[:,:,0],axis=1) < self.x_length ))
        self.eddy_locs = self.eddy_locs[keep_eddys]

        keep_eddys = np.where( (self.eddy_locs[:,1] + np.max(self.sigma_interp(self.eddy_locs[:,1])[:,:,1],axis=1) > 0.0 ))
        self.eddy_locs = self.eddy_locs[keep_eddys]
        keep_eddys = np.where( (self.eddy_locs[:,1] - np.max(self.sigma_interp(self.eddy_locs[:,1])[:,:,1],axis=1) < self.y_height ))
        self.eddy_locs = self.eddy_locs[keep_eddys]

        keep_eddys = np.where( (self.eddy_locs[:,2] + np.max(self.sigma_interp(self.eddy_locs[:,1])[:,:,2],axis=1) > 0.0 ))
        self.eddy_locs = self.eddy_locs[keep_eddys]
        keep_eddys = np.where( (self.eddy_locs[:,2] - np.max(self.sigma_interp(self.eddy_locs[:,1])[:,:,2],axis=1) < self.z_width  ))
        self.eddy_locs = self.eddy_locs[keep_eddys]

    @property
    def VB(self):
        #generate eddy volume
        if self.flow_type in ['channel', 'freeshear']:
            lows  = [          0.0 - self.sigma_x_max/2.0,           0.0 - self.sigma_y_max/2.0,          0.0 - self.sigma_z_max/2.0]
            highs = [self.x_length + self.sigma_x_max/2.0, self.y_height + self.sigma_y_max/2.0, self.z_width + self.sigma_z_max/2.0]
        elif self.flow_type == 'bl':
            lows  = [          0.0 - self.sigma_x_max/2.0,           0.0 - self.sigma_y_max/2.0,          0.0 - self.sigma_z_max/2.0]
            highs = [self.x_length + self.sigma_x_max/2.0,    self.delta + self.sigma_y_max/2.0, self.z_width + self.sigma_z_max/2.0]

        return np.product(np.array(highs) - np.array(lows))
