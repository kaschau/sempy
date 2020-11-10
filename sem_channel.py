import numpy as np
import matplotlib.pyplot as plt
import sys

from geometries.box import box
from generate_primes import generate_primes

#Flow
Re = 10e5
tme = 10 #Time of signal, used to determine length of box
utau = 0.1
delta = np.pi #Defined from flow configuration
viscosity = 1e-5
Ublk = Re*viscosity/delta #Bulk flow velocity, used to determine length of box
utau = Ublk/(5*np.log10(Re))
nframes = 200
#Box geobm
y_height = delta*2.0
z_width = 2*np.pi

#Eddy Density
C_Eddy = 1.0

#Initialize domain
domain = box('channel',Ublk,tme,y_height,z_width,delta,utau,viscosity)

#Set flow properties from existing data
domain.set_sem_data(sigmas_from='jarrin',stats_from='moser',profile_from='channel')

#Populate the domain
domain.populate(C_Eddy,'PDF')
#Create the eps
domain.generate_eps()
#Make it periodic
domain.make_periodic(periodic_x=True,periodic_y=True,periodic_z=True)
#Compute sigmas
domain.compute_sigmas()

domain.print_info()

#Create y,z coordinate pairs for calculation
ys = np.linspace(0.001*y_height,y_height*0.999,20)
zs = np.ones(ys.shape[0])*np.pi

#Compute u'
primes = generate_primes(ys,zs,domain,nframes,normalization='exact')

#Compute stats along line
uus = np.mean(primes[:,:,0]**2,axis=1)
vvs = np.mean(primes[:,:,1]**2,axis=1)
wws = np.mean(primes[:,:,2]**2,axis=1)

uvs = np.mean(primes[:,:,0]*primes[:,:,1],axis=1)
uws = np.mean(primes[:,:,0]*primes[:,:,2],axis=1)
vws = np.mean(primes[:,:,1]*primes[:,:,2],axis=1)

#Compute Ubars
Us = domain.Ubar_interp(ys) + np.mean(primes[:,:,0],axis=1)

#Compare signal to moser
#Ubar
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$\bar{U}$')
ax1.plot(Us,ys,label=r'$\bar{U}$ SEM', color='orange')
ax1.scatter(domain.Ubar_interp(ys),ys,label=r'$\bar{U}$ Profile',color='k')
ax1.legend()
plt.savefig('Ubar.png')
plt.close()

#Ruu
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uu}$')
ax1.plot(uus,ys,label=r'$R_{uu}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,0,0],ys,label=r'$R_{uu}$ Moser', color='k')
ax1.legend()
plt.savefig('Ruu.png')
plt.close()

#Rvv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{vv}$')
ax1.plot(vvs,ys,label=r'$R_{vv}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,1,1],ys,label=r'$R_{vv}$ Moser', color='k')
ax1.legend()
plt.savefig('Rvv.png')
plt.close()

#Rww
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{ww}$')
ax1.plot(wws,ys,label=r'$R_{ww}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,2,2],ys,label=r'$R_{ww}$ Moser', color='k')
ax1.legend()
plt.savefig('Rww.png')
plt.close()

#Ruv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uv}$')
ax1.plot(uvs,ys,label=r'$R_{uv}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,0,1],ys,label=r'$R_{uv}$ Moser', color='k')
ax1.legend()
plt.savefig('Ruv.png')
plt.close()

#Ruw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uw}$')
ax1.plot(uws,ys,label=r'$R_{uw}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,0,2],ys,label=r'$R_{uw}$ Moser', color='k')
ax1.legend()
plt.savefig('Ruw.png')
plt.close()

#Rvw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{vw}$')
ax1.plot(vws,ys,label=r'$R_{vw}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,1,2],ys,label=r'$R_{vw}$ Moser', color='k')
ax1.legend()
plt.savefig('Rvw.png')
plt.close()

#Sample u prime
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$u \prime$')
ax1.set_xlabel(r't')
ax1.set_title('Centerline Fluctiations')
ax1.plot(np.linspace(0,domain.x_length,nframes),primes[int((primes.shape[0])/2),:,0],color='orange')
plt.savefig('uprime.png')
plt.close()
