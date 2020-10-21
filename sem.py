import numpy as np
import matplotlib.pyplot as plt
import sys

from sigmas.jarrin_channel import sigma_interps
from stats.moser_channel import stat_interps

np.random.seed(10101)

#Flow
U0 = 10 #m/s
tme = 5 #s
u_tau = 0.1
delta = np.pi

#Box geom
x_length = U0*tme
y_height = 2*np.pi
z_width = 2*np.pi

#Eddy Density
C_Eddy = 1.0

#construct sigma interpolation functions
eddy_sigma_interp = sigma_interps(delta, u_tau)

#construct Rij and Ubar interpolation functins
Rij_interp,Ubar_interp = stat_interps(delta, u_tau)

#determine min,max sigmas
sigma_x_min = np.min(eddy_sigma_interp(0)[0])
sigma_x_max = np.max(eddy_sigma_interp(1.0)[0])

sigma_y_min = np.min(eddy_sigma_interp(0)[1])
sigma_y_max = np.max(eddy_sigma_interp(1.0)[1])

sigma_z_min = np.min(eddy_sigma_interp(0)[2])
sigma_z_max = np.max(eddy_sigma_interp(1.0)[2])

#generate box
lows  = [         - sigma_x_max,          - sigma_y_max,         - sigma_z_max]
highs = [x_length + sigma_x_max, y_height + sigma_y_max, z_width + sigma_z_max]
VB = np.product(np.array(highs) - np.array(lows))

#Compute number of eddys
neddy = int( C_Eddy * VB / (sigma_x_min*sigma_y_min*sigma_z_min) )
neddy = 10000
eddy_locs = np.random.uniform(low=lows,high=highs,size=(neddy,3))

#Compute eddy sigmas as function of y
eddy_sigmas = eddy_sigma_interp(eddy_locs[:,1])

#Compute Rij as a function of y
Rij = Rij_interp(eddy_locs[:,1])
#Cholesky decomp of all eddys
aij = np.linalg.cholesky(Rij)

#generate epsilons
eps_k = np.where(np.random.random((neddy,3,1)) <= 0.5,1.0,-1.0)

#####################################
######   Compute Fluctuations #######
#####################################

#Point on inlet face
y,z = y_height/2.0, z_width/2.0

#Compute Ubar as a function of y
Ubar = Ubar_interp(y)

#Define "time" point
xs = np.linspace(0,x_length,200)
#xs = np.array([0.0])
t = xs/U0
primes = np.empty((len(xs),3))

line_eddy_contributing = []
#Find eddies that contribute on line for u'
line_eddy_contributing.append(np.where( (np.abs( eddy_locs[:,1] - y ) < eddy_sigmas[:,0,1] )
                                      & (np.abs( eddy_locs[:,2] - z ) < eddy_sigmas[:,0,2] ) ) )

#Find eddies that contribute on line for v'
line_eddy_contributing.append(np.where( (np.abs( eddy_locs[:,1] - y ) < eddy_sigmas[:,1,1] )
                                      & (np.abs( eddy_locs[:,2] - z ) < eddy_sigmas[:,1,2] ) ) )

#Find eddies that contribute on line for w'
line_eddy_contributing.append(np.where( (np.abs( eddy_locs[:,1] - y ) < eddy_sigmas[:,2,1] )
                                      & (np.abs( eddy_locs[:,2] - z ) < eddy_sigmas[:,2,2] ) ) )

for i,x in enumerate(xs):
    print('Computing t = ',x)
    for k in range(3):
        #Find all non zero eddies for u_k'
        on_line = line_eddy_contributing[k]
        x_dist = np.abs( eddy_locs[on_line][:,0] - x )
        point_eddy_contributing = np.where( x_dist < eddy_sigmas[on_line][:,k,0] )

        x_dist = np.abs( eddy_locs[on_line][point_eddy_contributing][:,0] - x )
        y_dist = np.abs( eddy_locs[on_line][point_eddy_contributing][:,1] - y )
        z_dist = np.abs( eddy_locs[on_line][point_eddy_contributing][:,2] - z )

        x_sigma = eddy_sigmas[on_line][point_eddy_contributing][:,k,0]
        y_sigma = eddy_sigmas[on_line][point_eddy_contributing][:,k,1]
        z_sigma = eddy_sigmas[on_line][point_eddy_contributing][:,k,2]

        #Individual f(x)
        fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
        fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
        fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

        fx  = np.sqrt(VB)/np.sqrt(np.product((x_sigma,y_sigma,z_sigma),axis=0)) * fxx*fxy*fxz

        A = aij[on_line][point_eddy_contributing]
        e = eps_k[on_line][point_eddy_contributing]
        ck = np.matmul(A, e).reshape(e.shape[0:2])

        Xk = ck*fx.reshape(fx.shape[0],1)

        prime = 1.0/np.sqrt(neddy) * np.sum( Xk ,axis=0)

        primes[i,k] = prime[k]

plt.plot(t,Ubar+primes[:,0])
plt.plot(t,primes[:,1])
plt.plot(t,primes[:,2])
plt.show()
