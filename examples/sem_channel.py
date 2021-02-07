import numpy as np
import matplotlib.pyplot as plt
import sempy

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
C_Eddy = 2.0

#eddy convection speed
convect='local'

#Initialize domain
domain = sempy.geometries.box('channel',Ublk,tme,y_height,z_width,delta,utau,viscosity)

#Set flow properties from existing data
domain.set_sem_data(sigmas_from='jarrin',stats_from='moser',profile_from='channel')

#Populate the domain
domain.populate(C_Eddy,'PDF',convect=convect)
#Create the eps
domain.generate_eps()
#Make it periodic
domain.make_periodic(periodic_x=False,periodic_y=False,periodic_z=True,convect=convect)
#Compute sigmas
domain.compute_sigmas()

domain.print_info()

#Create y,z coordinate pairs for calculation
ys = np.concatenate((np.linspace(0,0.01*domain.delta,5),
                     np.linspace(0.01*domain.delta,1.99*domain.delta,20),
                     np.linspace(1.99*domain.delta,2.0*domain.delta,5)))

zs = np.ones(ys.shape[0])*np.pi

#Compute u'
up,vp,wp = sempy.generate_primes(ys,zs,domain,nframes,normalization='exact',convect=convect)

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
ax1.scatter(domain.Ubar_interp(ys),ys,label=r'$\bar{U}$ Profile',color='k')
ax1.legend()
plt.savefig('Ubar.png')
plt.close()

#Ruu
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$R_{uu}$')
ax1.plot(uus,ys,label=r'$R_{uu}$ SEM', color='orange')
ax1.scatter(domain.Rij_interp(ys)[:,0,0],ys,label=r'$R_{uu}$ Theory', color='k')
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
ax1.plot(np.linspace(0,domain.x_length,nframes),up[:,int((up.shape[1])/2)],color='orange')
plt.savefig('uprime.png')
plt.close()
