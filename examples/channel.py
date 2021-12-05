import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sempy

"""
Reproduction of Moser Channel DNS at Re_{\tau} = 590
"""
# Flow
Re_tau = 587.19
delta = 0.05  # Made this up
viscosity = 1.81e-5  # Air @ STP
utau = Re_tau * viscosity / delta
Ublk = 2.12630000e01 * utau
tme = 10 * 2 * np.pi * delta / Ublk  # Based on FTT of DNS domain
nframes = 200
# Box geom
yHeight = 2.0 * delta
zWidth = np.pi * delta

# Eddy Density
cEddy = 1.0
# eddy convection speed
convect = "uniform"
# population method
popMeth = "random"
# normalization
norm = "exact"
# shape funcion
shape = "tent"

# Initialize domain
domain = sempy.geometries.box(
    "channel", Ublk, tme, yHeight, zWidth, delta, utau, viscosity
)
# Set flow properties from existing data
domain.setSemData(sigmasFrom="jarrin", statsFrom="moser", profileFrom="channel")
# Populate the domain
domain.populate(cEddy, method=popMeth)
# Create the eps
domain.generateEps()
# Compute sigmas
domain.computeSigmas()
# Make it periodic
domain.makePeriodic(periodicX=False, periodicY=False, periodicZ=True)

print(domain)

# Create y,z coordinate pairs for calculation
ys = np.concatenate(
    (
        np.linspace(0.0001 * domain.delta, 0.01 * domain.delta, 5),
        np.linspace(0.01 * domain.delta, 1.99 * domain.delta, 20),
        np.linspace(1.99 * domain.delta, 1.9999 * domain.delta, 5),
    )
)
zs = np.ones(ys.shape[0]) * zWidth / 2.0

# Compute u'
up, vp, wp = sempy.generatePrimes(
    ys, zs, domain, nframes, normalization=norm, convect=convect, shape=shape
)

# Compute stats along line
uus = np.mean(up[:, :] ** 2, axis=0)
vvs = np.mean(vp[:, :] ** 2, axis=0)
wws = np.mean(wp[:, :] ** 2, axis=0)

uvs = np.mean(up[:, :] * vp[:, :], axis=0)
uws = np.mean(up[:, :] * wp[:, :], axis=0)
vws = np.mean(vp[:, :] * wp[:, :], axis=0)

# Compute Ubars
Us = domain.ubarInterp(ys) + np.mean(up[:, :], axis=0)

# Compare signal to moser
# Make some plots
if matplotlib.checkdep_usetex(True):
    plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (6, 4.5)
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 1.0

# Ubar
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$\bar{U}$")
ax1.plot(Us, ys, label=r"$\bar{U}$ SEM", color="orange")
ax1.scatter(domain.ubarInterp(ys), ys, label=r"$\bar{U}$ Profile", color="k")
ax1.legend()
plt.savefig("Ubar.png")
plt.close()

# Ruu
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{uu}$")
ax1.plot(uus, ys, label=r"$R_{uu}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 0, 0], ys, label=r"$R_{uu}$ Moser", color="k")
ax1.legend()
plt.savefig("Ruu.png")
plt.close()

# Rvv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{vv}$")
ax1.plot(vvs, ys, label=r"$R_{vv}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 1, 1], ys, label=r"$R_{vv}$ Moser", color="k")
ax1.legend()
plt.savefig("Rvv.png")
plt.close()

# Rww
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{ww}$")
ax1.plot(wws, ys, label=r"$R_{ww}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 2, 2], ys, label=r"$R_{ww}$ Moser", color="k")
ax1.legend()
plt.savefig("Rww.png")
plt.close()

# Ruv
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{uv}$")
ax1.plot(uvs, ys, label=r"$R_{uv}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 0, 1], ys, label=r"$R_{uv}$ Moser", color="k")
ax1.legend()
plt.savefig("Ruv.png")
plt.close()

# Ruw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{uw}$")
ax1.plot(uws, ys, label=r"$R_{uw}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 0, 2], ys, label=r"$R_{uw}$ Moser", color="k")
ax1.legend()
plt.savefig("Ruw.png")
plt.close()

# Rvw
fig, ax1 = plt.subplots()
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$R_{vw}$")
ax1.plot(vws, ys, label=r"$R_{vw}$ SEM", color="orange")
ax1.scatter(domain.rijInterp(ys)[:, 1, 2], ys, label=r"$R_{vw}$ Moser", color="k")
ax1.legend()
plt.savefig("Rvw.png")
plt.close()

# Sample u prime
plt.rcParams["figure.figsize"] = (4, 6.5)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
ax1.set_ylabel(r"$u^{\prime}$")
ax1.grid(linestyle="--")
ax1.set_title("Centerline Fluctiations")
ax1.plot(
    np.linspace(0, domain.xLength / domain.Ublk, nframes),
    up[:, int((up.shape[1]) / 2)],
    color="orange",
)

ax2.set_ylabel(r"$v^{\prime}$")
ax2.grid(linestyle="--")
ax2.plot(
    np.linspace(0, domain.xLength / domain.Ublk, nframes),
    vp[:, int((vp.shape[1]) / 2)],
    color="blue",
)

ax3.set_ylabel(r"$w^{\prime}$")
ax3.grid(linestyle="--")
ax3.set_xlabel(r"$Time$")

ax3.plot(
    np.linspace(0, domain.xLength / domain.Ublk, nframes),
    wp[:, int((wp.shape[1]) / 2)],
    color="green",
)
plt.savefig("uprime.png")
plt.close()
