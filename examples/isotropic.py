import numpy as np
import matplotlib.pyplot as plt
import sempy
'''
A reproduction of the data from section 4.4 of Jarrin's thesis in which isotropic turbulence is produced.
'''
#Flow
tme = 60  #Time of signal, used to determine length of box
Ublk = 10 #Bulk flow velocity, used to determine length of box
delta = None
viscosity = 1e-5
utau = None
nframes = 10000
#Box geobm
y_height = 2.0*np.pi
z_width = y_height

#Eddy Density
C_Eddy = 2.0

#eddy convection speed
convect='uniform'

#normalization
normalization='jarrin'

#population method
pop_meth = 'random'

#Initialize domain
domain = sempy.geometries.box('shearfree',Ublk,tme,y_height,z_width,delta,utau,viscosity)

#Set flow properties from existing data
domain.set_sem_data(sigmas_from='uniform',stats_from='isotropic',profile_from='uniform',scale_factor=0.5)

#Populate the domain
domain.populate(C_Eddy,method=pop_meth,convect=convect)
#Create the eps
domain.generate_eps()
#Make it periodic
domain.make_periodic(periodic_x=False,periodic_y=False,periodic_z=False)
#Compute sigmas
domain.compute_sigmas()

domain.print_info()

#Create y,z coordinate pairs for calculation
ys = np.array([np.pi])
zs = np.ones(ys.shape[0])*y_height

#Compute u'
up,vp,wp = sempy.generate_primes(ys,zs,domain,nframes,normalization=normalization)

#Compute stats along line
uus = np.mean(up[:,:]**2,axis=0)
vvs = np.mean(vp[:,:]**2,axis=0)
wws = np.mean(wp[:,:]**2,axis=0)

uvs = np.mean(up[:,:]*vp[:,:],axis=0)
uws = np.mean(up[:,:]*wp[:,:],axis=0)
vws = np.mean(vp[:,:]*wp[:,:],axis=0)

#Compute Ubars
Us = domain.Ubar_interp(ys) + np.mean(up[:,:],axis=0)

#Sample u prime
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$u \prime$')
ax1.set_xlabel(r't')
ax1.set_title('Centerline Fluctiations')
x = np.linspace(0,domain.x_length/domain.Ublk,nframes)
xind = np.where(x <= 8)
ax1.plot(x[xind],10+up[:,int((up.shape[1])/2)][xind],color='orange')
plt.savefig('uprime.png')
plt.close()


u=up[:,int((up.shape[1])/2)]
v=vp[:,int((up.shape[1])/2)]
w=wp[:,int((up.shape[1])/2)]
uu = []
vv = []
ww = []
uv = []
uw = []
vw = []
for i in range(len(u)-2):
    uu.append(np.mean(u[1:i+2]**2))
    vv.append(np.mean(v[1:i+2]**2))
    ww.append(np.mean(w[1:i+2]**2))
    uv.append(np.mean(u[1:i+2]*v[1:i+2]))
    uw.append(np.mean(u[1:i+2]*w[1:i+2]))
    vw.append(np.mean(v[1:i+2]*w[1:i+2]))

#Convergence
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'$<u_i u_j>$')
ax1.set_xlabel(r't')
x = np.linspace(0,domain.x_length/domain.Ublk,nframes)[2::]
ax1.plot(x,uu,color='orange',label='uu')
ax1.plot(x,vv,color='red',label='vv')
ax1.plot(x,ww,color='gold',label='ww')
ax1.plot(x,uv,color='blue',label='uv')
ax1.plot(x,uw,color='green',label='uw')
ax1.plot(x,vw,color='k',label='vw')
ax1.legend()
plt.savefig('converge.png')
plt.close()
