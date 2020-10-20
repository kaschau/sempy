import numpy as np
import matplotlib.pyplot as plt
import sys

from sigmas import jarrin_SST

np.random.seed(10101)

#Flow
U0 = 10 #m/s
tme = 8 #s
u_tau = 1

#Box geom
x_length = U0*tme
y_height = 2*np.pi
z_width = 2*np.pi

#construct sigma interpolation functions
eddy_sigmas = np.ones(3,3,eddy_locs.shape[0])
eddy_sigmas[:,0] = eddy_l_x
eddy_sigmas[:,1] = eddy_l_y
eddy_sigmas[:,2] = eddy_l_z


#generate box of eddys
lows  = [         - eddy_l_x,          - eddy_l_y,         - eddy_l_z]
highs = [x_length + eddy_l_x, y_height + eddy_l_y, z_width + eddy_l_z]
VB = np.product(np.array(highs) - np.array(lows))

eddy_density = 1
neddy = int( eddy_density * VB / (eddy_l_x*eddy_l_y*eddy_l_z) )
eddy_locs = np.random.uniform(low=lows,high=highs,size=(neddy,3))


#generate epsilons
eps_k = np.where(np.random.random(eddy_locs.shape) <= 0.5,1.0,-1.0)

#Define "time" point
y,z = y_height/2.0, z_width/2.0
xs = np.linspace(0,x_length,2000)
#xs = np.array([0.0])
t = xs/U0
primes = np.empty((len(xs),3))

#Find eddies that contribute on line
line_eddy_contributing = np.where( (np.abs( eddy_locs[:,1] - y ) < eddy_sigmas[:,1] )
                                 & (np.abs( eddy_locs[:,2] - z ) < eddy_sigmas[:,2] ) )

#construct Rij
Rij = np.array([ [1.0,0.0,0.0] , [0.0,1.0,0.0] , [0.0,0.0,1.0] ])
aij = np.linalg.cholesky(Rij)

for i,x in enumerate(xs):
    #Find all non zero eddies
    x_dist = np.abs( eddy_locs[line_eddy_contributing][:,0] - x )
    point_eddy_contributing = np.where( x_dist < eddy_sigmas[line_eddy_contributing][:,0] )

    x_dist = np.abs( eddy_locs[line_eddy_contributing][point_eddy_contributing][:,0] - x )
    y_dist = np.abs( eddy_locs[line_eddy_contributing][point_eddy_contributing][:,1] - y )
    z_dist = np.abs( eddy_locs[line_eddy_contributing][point_eddy_contributing][:,2] - z )

    x_sigma = eddy_sigmas[line_eddy_contributing][point_eddy_contributing][:,0]
    y_sigma = eddy_sigmas[line_eddy_contributing][point_eddy_contributing][:,1]
    z_sigma = eddy_sigmas[line_eddy_contributing][point_eddy_contributing][:,2]

    #Individual f(x)
    fxx = np.sqrt(1.5)*(1.0-x_dist/x_sigma)
    fxy = np.sqrt(1.5)*(1.0-y_dist/y_sigma)
    fxz = np.sqrt(1.5)*(1.0-z_dist/z_sigma)

    fx  = np.sqrt(VB)/np.sqrt(np.product((x_sigma,y_sigma,z_sigma),axis=0)) * fxx*fxy*fxz

    ck = np.matmul(aij,eps_k[line_eddy_contributing][point_eddy_contributing].T).T
    Xk = ck*fx.reshape(fx.shape[0],1)

    prime = 1.0/np.sqrt(neddy) * np.sum( Xk ,axis=0)

    primes[i,:] = prime

plt.plot(t,10+primes[:,0])
plt.show()
