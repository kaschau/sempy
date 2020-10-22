import numpy as np
import matplotlib.pyplot as plt
import sys

from sigmas.jarrin_channel import create_sigma_interps
from stats.moser_channel import create_stat_interps

np.random.seed(1010)

#Flow
U0 = 10 #Bulk flow velocity, used to determine length of box
tme = 1 #Time of signal, used to determine length of box
u_tau = 0.1
delta = np.pi #Defined from flow configuration

#Box geom
y_height = 2*np.pi
z_width = 2*np.pi
x_length = U0*tme

#Eddy Density
C_Eddy = 1.0

#construct sigma interpolation functions
eddy_sigma_interp = create_sigma_interps(delta, u_tau)

#construct Rij and Ubar interpolation functins
Rij_interp,Ubar_interp = create_stat_interps(delta, u_tau)

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

#Generate random eddy locations
eddy_locs = np.random.uniform(low=lows,high=highs,size=(neddy,3))

#Compute all eddy sigmas as function of y
eddy_sigmas = eddy_sigma_interp(eddy_locs[:,1])

#Compute Rij for all eddys as a function of y
Rij = Rij_interp(eddy_locs[:,1])
#Cholesky decomp of all eddys
aij = np.linalg.cholesky(Rij)

#generate epsilons
eps_k = np.where(np.random.random((neddy,3,1)) <= 0.5,1.0,-1.0)

#####################################
######   Compute Fluctuations #######
#####################################

#Generate points on inlet face
z = z_width/2.0
ys = np.linspace(y_height*0.001,y_height*0.999,50)

# Some post processing storage
Us = [];uus = [];vvs = [];wws = []
uvs = [];uws = [];vws = []

#Loop over each location on face
for y in ys:
    print('Computing y = ',y)

    #Compute local Ubar as a function of y
    Ubar = Ubar_interp(y)

    #Define "time" points for frames
    xs = np.linspace(0,x_length,100)

    #Storage for fluctuations
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

    Us.append ( Ubar + np.mean(primes[:,0]   ) )
    uus.append( np.mean(primes[:,0]**2) )
    vvs.append( np.mean(primes[:,1]**2) )
    wws.append( np.mean(primes[:,2]**2) )

    uvs.append(np.mean(primes[:,0]*primes[:,1]) )
    uws.append(np.mean(primes[:,0]*primes[:,2]) )
    vws.append(np.mean(primes[:,1]*primes[:,2]) )


#Compare signal to moser
#Ubar
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$\bar{U}$')
ax1.plot(Us,ys,label=r'$\bar{U}$ SEM')
ax1.plot(Ubar_interp(ys),ys,label=r'$\bar{U}$ Moser')
ax1.legend()
plt.savefig('Ubar.png')
plt.close()

#Ruu
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uu}$')
ax1.plot(uus,ys,label=r'$R_{uu}$ SEM')
ax1.plot(Rij_interp(ys)[:,0,0],ys,label=r'$R_{uu}$ Moser')
ax1.legend()
plt.savefig('Ruu.png')
plt.close()

#Rvv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{vv}$')
ax1.plot(vvs,ys,label=r'$R_{vv}$ SEM')
ax1.plot(Rij_interp(ys)[:,1,1],ys,label=r'$R_{vv}$ Moser')
ax1.legend()
plt.savefig('Rvv.png')
plt.close()

#Rww
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{ww}$')
ax1.plot(wws,ys,label=r'$R_{ww}$ SEM')
ax1.plot(Rij_interp(ys)[:,2,2],ys,label=r'$R_{ww}$ Moser')
ax1.legend()
plt.savefig('Rww.png')
plt.close()

#Ruv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uv}$')
ax1.plot(uvs,ys,label=r'$R_{uv}$ SEM')
ax1.plot(Rij_interp(ys)[:,0,1],ys,label=r'$R_{uv}$ Moser')
ax1.legend()
plt.savefig('Ruv.png')
plt.close()

#Ruw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uw}$')
ax1.plot(uws,ys,label=r'$R_{uw}$ SEM')
ax1.plot(Rij_interp(ys)[:,0,2],ys,label=r'$R_{uw}$ Moser')
ax1.legend()
plt.savefig('Ruw.png')
plt.close()

#Rvw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{vw}$')
ax1.plot(vws,ys,label=r'$R_{vw}$ SEM')
ax1.plot(Rij_interp(ys)[:,1,2],ys,label=r'$R_{vw}$ Moser')
ax1.legend()
plt.savefig('Rvw.png')
plt.close()
