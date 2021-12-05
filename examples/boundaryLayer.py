import numpy as np
import matplotlib.pyplot as plt
import sempy

# Flow
Uo = 10  # Bulk flow velocity, used to determine length of box
tme = 10  # Time of signal, used to determine length of box
utau = 0.1
delta = 1.0  # Defined from flow configuration
viscosity = 1e-5
nframes = 200
# Box geobm
yHeight = 1.2 * delta
zWidth = 2 * delta

# Eddy Density
cEddy = 1.0

# Initialize domain as 'bl'
domain = sempy.geometries.box("bl", Uo, tme, yHeight, zWidth, delta, utau, viscosity)

# Set flow properties from existing data
domain.setSemData(sigmasFrom="linear_bl", statsFrom="spalart", profileFrom="bl")

# Populate the domain
domain.populate(cEddy, "PDF")
# Create the eps
domain.generateEps()
# Compute sigmas
domain.computeSigmas()
# Make it periodic
domain.makePeriodic(periodicX=True, periodicZ=True)

print(domain)

# Create y,z coordinate pairs for calculation
ys = np.linspace(0.001 * yHeight, yHeight * 0.999, 10)
zs = np.ones(ys.shape[0]) * zWidth / 2.0

# Compute u'
up, vp, wp = sempy.generatePrimes(
    ys, zs, domain, nframes, normalization="exact", convect="uniform"
)

# Compute stats along line
uus = np.mean(up[:, :] ** 2, axis=0)
vvs = np.mean(vp[:, :] ** 2, axis=0)
wws = np.mean(wp[:, :] ** 2, axis=0)

uvs = np.mean(up[:, :] * vp[:, :], axis=0)
uws = np.mean(up[:, :] * wp[:, :], axis=0)
vws = np.mean(vp[:, :] * wp[:, :], axis=0)

# Compute Ubars
Us = domain.Ubar_interp(ys) + np.mean(up[:, :], axis=0)

# Compare signal to moser
# Ubar
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$\bar{U}$")
ax1.plot(Us, ys, label=r"$\bar{U}$ SEM", color="orange")
ax1.scatter(domain.ubarInterp(ys), ys, label=r"$\bar{U}$ Theory", color="k")
ax1.legend()
plt.savefig("Ubar.png")
plt.close()

# Ruu
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{uu}$")
ax1.plot(uus, ys, label=r"$R_{uu}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 0, 0], ys, label=r"$R_{uu}$ Spalart", color="k")
ax1.legend()
plt.savefig("Ruu.png")
plt.close()

# Rvv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{vv}$")
ax1.plot(vvs, ys, label=r"$R_{vv}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 1, 1], ys, label=r"$R_{vv}$ Spalart", color="k")
ax1.legend()
plt.savefig("Rvv.png")
plt.close()

# Rww
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{ww}$")
ax1.plot(wws, ys, label=r"$R_{ww}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 2, 2], ys, label=r"$R_{ww}$ Spalart", color="k")
ax1.legend()
plt.savefig("Rww.png")
plt.close()

# Ruv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{uv}$")
ax1.plot(uvs, ys, label=r"$R_{uv}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 0, 1], ys, label=r"$R_{uv}$ Spalart", color="k")
ax1.legend()
plt.savefig("Ruv.png")
plt.close()

# Ruw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{uw}$")
ax1.plot(uws, ys, label=r"$R_{uw}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 0, 2], ys, label=r"$R_{uw}$ Spalart", color="k")
ax1.legend()
plt.savefig("Ruw.png")
plt.close()

# Rvw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{vw}$")
ax1.plot(vws, ys, label=r"$R_{vw}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 1, 2], ys, label=r"$R_{vw}$ Spalart", color="k")
ax1.legend()
plt.savefig("Rvw.png")
plt.close()

# Sample u prime
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$u \prime$")
ax1.set_xlabel(r"t")
ax1.set_title("Free Stream Fluctuations")
ax1.plot(np.linspace(0, domain.xLength, nframes), up[:, -1], color="orange")
plt.savefig("uprime.png")
plt.close()
