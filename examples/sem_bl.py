import numpy as np
import matplotlib.pyplot as plt
import sempy

#Flow
Ublk = 10 #Bulk flow velocity, used to determine length of box
tme = 10 #Time of signal, used to determine length of box
utau = 0.1
delta = 1.0 #Defined from flow configuration
viscosity = 1e-5
nframes = 200
#Box geobm
y_height = 1.2*delta
z_width = 2*delta

#Eddy Density
C_Eddy = 1.0

#Initialize domain as 'bl'
domain = sempy.geometries.box('bl',Ublk,tme,y_height,z_width,delta,utau,viscosity)

#Set flow properties from existing data
domain.set_sem_data(sigmas_from='linear_bl',stats_from='spalart',profile_from='bl')

#Populate the domain
domain.populate(C_Eddy,'PDF')
#Create the eps
domain.generate_eps()
#Compute sigmas
domain.compute_sigmas()
#Make it periodic
domain.make_periodic(periodic_x=True,periodic_z=True)

domain.print_info()

#Create y,z coordinate pairs for calculation
ys = np.linspace(0.001*y_height,y_height*0.999,10)
zs = np.ones(ys.shape[0])*z_width/2.0

#Compute u'
up,vp,wp = sempy.generate_primes(ys,zs,domain,nframes,normalization='exact')

#Compute stats along line
uus = np.mean(up[:,:]**2,axis=0)
vvs = np.mean(vp[:,:]**2,axis=0)
wws = np.mean(wp[:,:]**2,axis=0)

uvs = np.mean(up[:,:]*vp[:,:],axis=0)
uws = np.mean(up[:,:]*wp[:,:],axis=0)
vws = np.mean(vp[:,:]*wp[:,:],axis=0)

#Compute Ubars
Us = domain.Ubar_interp(ys) + np.mean(up[:,:],axis=0)

#Compare signal to moser
#Ubar
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$\bar{U}$')
ax1.plot(Us,ys,label=r'$\bar{U}$ SEM', color='orange')
ax1.scatter(domain.Ubar_interp(ys),ys,label=r'$\bar{U}$ Theory',color='k')
ax1.legend()
plt.savefig('Ubar.png')
plt.close()

#Ruu
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uu}$')
ax1.plot(uus,ys,label=r'$R_{uu}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,0,0],ys,label=r'$R_{uu}$ Spalart', color='k')
ax1.legend()
plt.savefig('Ruu.png')
plt.close()

#Rvv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{vv}$')
ax1.plot(vvs,ys,label=r'$R_{vv}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,1,1],ys,label=r'$R_{vv}$ Spalart', color='k')
ax1.legend()
plt.savefig('Rvv.png')
plt.close()

#Rww
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{ww}$')
ax1.plot(wws,ys,label=r'$R_{ww}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,2,2],ys,label=r'$R_{ww}$ Spalart', color='k')
ax1.legend()
plt.savefig('Rww.png')
plt.close()

#Ruv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uv}$')
ax1.plot(uvs,ys,label=r'$R_{uv}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,0,1],ys,label=r'$R_{uv}$ Spalart', color='k')
ax1.legend()
plt.savefig('Ruv.png')
plt.close()

#Ruw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uw}$')
ax1.plot(uws,ys,label=r'$R_{uw}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,0,2],ys,label=r'$R_{uw}$ Spalart', color='k')
ax1.legend()
plt.savefig('Ruw.png')
plt.close()

#Rvw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{vw}$')
ax1.plot(vws,ys,label=r'$R_{vw}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,1,2],ys,label=r'$R_{vw}$ Spalart', color='k')
ax1.legend()
plt.savefig('Rvw.png')
plt.close()

#Sample u prime
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$u \prime$')
ax1.set_xlabel(r't')
ax1.set_title('Free Stream Fluctuations')
ax1.plot(np.linspace(0,domain.x_length,nframes),up[:,-1],color='orange')
plt.savefig('uprime.png')
plt.close()
