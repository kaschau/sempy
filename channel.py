import numpy as np
import scipy.integrate as integrate
import sys
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

    def set_flow_data(self,sigmas_from='jarrin',stats_from='moser',rsigma=1.0):

        if sigmas_from == 'jarrin':
            from sigmas.jarrin_channel import create_sigma_interps
        elif sigmas_from == 'uniform':
            from sigmas.uniform import create_sigma_interps

        if stats_from == 'moser':
            from stats.moser_channel import create_stat_interps

        self.sigma_interp = create_sigma_interps(self.delta,self.utau,self.ymin,rsigma)
        self.Rij_interp,self.Ubar_interp = create_stat_interps(self.delta,self.utau,self.ymin)


    def populate_random(self, C_Eddy=1.0, periodic_x=True,periodic_y=False,periodic_z=False):

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

        V_sigma_min = np.min(np.product(test_sigmas,axis=1))

        #generate eddy volume
        lows  = [self.xmin - sigma_x_max, self.ymin - sigma_y_max, self.zmin - sigma_z_max]
        highs = [self.xmax + sigma_x_max, self.ymax + sigma_y_max, self.zmax + sigma_z_max]

        self.VB = np.product(np.array(highs) - np.array(lows))

        #Compute number of eddys
        neddy = int( C_Eddy * self.VB / V_sigma_min )

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
            periodic_eddys[:,2] = periodic_eddys[:,2] + self.zmax
            temp_eddy_locs = np.concatenate( (temp_eddy_locs, periodic_eddys) )

        self.eddy_locs = temp_eddy_locs
        self.neddy = self.eddy_locs.shape[0]

        #Compute all eddy sigmas as function of y
        self.sigmas = self.sigma_interp(self.eddy_locs[:,1])

        #generate epsilons
        temp_eps = np.where(np.random.uniform(low=-1,high=1,size=(self.neddy,3))<= 0.0, -1.0,1.0)
        #Transfer periodic epsilons
        if periodic_x:
            temp_eps[np.where(temp_eddy_locs[:,0] > self.xmax-sigma_x_max)] = temp_eps[np.where(self.eddy_locs[:,0] < self.xmin+sigma_x_max)]
        if periodic_y:
            temp_eps[np.where(temp_eddy_locs[:,1] > self.ymax-sigma_y_max)] = temp_eps[np.where(self.eddy_locs[:,1] < self.ymin+sigma_y_max)]
        if periodic_z:
            temp_eps[np.where(temp_eddy_locs[:,2] > self.zmax-sigma_z_max)] = temp_eps[np.where(self.eddy_locs[:,2] < self.zmin+sigma_z_max)]

        self.eps = temp_eps

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

    def populate_PDF(self, C_Eddy=1.0, periodic_x=True,periodic_y=False,periodic_z=False):

        if not hasattr(self,'sigma_interp'):
            raise ValueError('Please set your flow data before trying to populate your domain')

        #determine min,max sigmas
        test_ys = np.linspace(self.ymin,self.ymax,200)
        test_sigmas = self.sigma_interp(test_ys)

        sigma_x_min = np.min(test_sigmas[:,:,0])
        sigma_x_max = np.max(test_sigmas[:,:,0])
        sigma_y_min = np.min(test_sigmas[:,:,1])
        sigma_y_max = np.max(test_sigmas[:,:,1])
        sigma_z_min = np.min(test_sigmas[:,:,2])
        sigma_z_max = np.max(test_sigmas[:,:,2])

        #generate eddy volume
        lows  = [self.xmin - sigma_x_max, self.ymin - sigma_y_max, self.zmin - sigma_z_max]
        highs = [self.xmax + sigma_x_max, self.ymax + sigma_y_max, self.zmax + sigma_z_max]

        self.VB = np.product(np.array(highs) - np.array(lows))

        #Eddy volumes as a function of y
        test_ys = np.linspace(self.ymin-sigma_y_max, self.ymax+sigma_y_max,200)
        test_sigmas = self.sigma_interp(test_ys)
        #Smallest eddy y height
        # V_eddy = np.min(np.product(test_sigmas,axis=1),axis=1)
        Y_eddy = np.min(test_sigmas[:,:,1], axis=1)

        Y_eddy_min = Y_eddy.min()
        Y_eddy_max = Y_eddy.max()
        Y_ratio = Y_eddy_max/Y_eddy_min
        #Flip and shift so largest eddy sits on zero
        Y_eddy_prime = - ( Y_eddy - Y_eddy_min ) + (Y_eddy_max-Y_eddy_min)

        #Rescale so lowest point is equal to one
        Y_eddy_prime = Y_eddy_prime/Y_eddy_max*Y_ratio + 1.0
        Y_eddy_norm = integrate.trapz(Y_eddy_prime,test_ys)
        #Create a PDF of eddy placement in y
        pdf = Y_eddy_prime/Y_eddy_norm
        expected_Veddy = integrate.trapz(pdf*Y_eddy,test_ys)**3

        # import matplotlib.pyplot as plt
        # plt.plot(test_ys,Y_eddy)
        # plt.plot(test_ys,np.ones(200)*expected_Veddy)
        # plt.show()
        # plt.plot(test_ys,pdf)
        # plt.show()
        # sys.exit()
        neddy = int( C_Eddy * self.VB / expected_Veddy )

        #Create bins for placing eddys in y
        dy = np.abs(test_ys[1]-test_ys[0])
        bin_centers = 0.5*( test_ys[1::] + test_ys[0:-1] )
        bin_pdf = 0.5*( pdf[1::] + pdf[0:-1] ) * dy
        #place neddys in bins according to pdf
        bin_loc = np.random.choice(np.arange(bin_pdf.shape[0]),p=bin_pdf,size=neddy)

        #create y values for eddys
        eddy_ys = np.take(bin_centers, bin_loc) + np.random.uniform(low=-dy/2.0,high=dy/2.0,size=neddy)

        #Generate random eddy locations
        temp_eddy_locs = np.random.uniform(low=lows,high=highs,size=(neddy,3))
        #Replace ys with PDF ys
        temp_eddy_locs[:,1] = eddy_ys

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
            periodic_eddys[:,2] = periodic_eddys[:,2] + self.zmax
            temp_eddy_locs = np.concatenate( (temp_eddy_locs, periodic_eddys) )

        self.eddy_locs = temp_eddy_locs
        self.neddy = self.eddy_locs.shape[0]

        #Compute all eddy sigmas as function of y
        self.sigmas = self.sigma_interp(self.eddy_locs[:,1])

        #generate epsilons
        temp_eps = np.where(np.random.uniform(low=-1,high=1,size=(self.neddy,3))<= 0.0, -1.0,1.0)
        #Transfer periodic epsilons
        if periodic_x:
            temp_eps[np.where(temp_eddy_locs[:,0] > self.xmax-sigma_x_max)] = temp_eps[np.where(self.eddy_locs[:,0] < self.xmin+sigma_x_max)]
        if periodic_y:
            temp_eps[np.where(temp_eddy_locs[:,1] > self.ymax-sigma_y_max)] = temp_eps[np.where(self.eddy_locs[:,1] < self.ymin+sigma_y_max)]
        if periodic_z:
            temp_eps[np.where(temp_eddy_locs[:,2] > self.zmax-sigma_z_max)] = temp_eps[np.where(self.eddy_locs[:,2] < self.zmin+sigma_z_max)]

        self.eps = temp_eps

        print(f'Eddy box volume of {self.VB}')
        print(f'Generating {self.neddy} eddys')
        print('Using:')
        print(f'      sigma_x_min = {sigma_x_min}')
        print(f'      sigma_x_max = {sigma_x_max}')
        print(f'      sigma_y_min = {sigma_y_min}')
        print(f'      sigma_y_max = {sigma_y_max}')
        print(f'      sigma_z_min = {sigma_z_min}')
        print(f'      sigma_z_max = {sigma_z_max}')
