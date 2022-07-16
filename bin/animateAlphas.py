"""
This utility plots the results of semForPEREGRINE by plotting the u,v,w
components at the values of the frames, and creates animation
videos of the results.

Uses identical input file that was used in semForPEREGRINE. Expects the alphas
need to be located in ./<semForPEREGRINE.inp>_alphas

Creats three videos, U.mp4, v.mp4, and w.mp4

Example
-------
/path/to/sempy/utilities/animateAlphas.py <sem.inp>

"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import peregrinepy as pg
import yaml
from matplotlib.animation import FuncAnimation


def getSlice(fn):
    if fn == 1:
        return np.s_[0, :, :]
    elif fn == 2:
        return np.s_[-1, :, :]
    elif fn == 3:
        return np.s_[:, 0, :]
    elif fn == 4:
        return np.s_[:, -1, :]
    elif fn == 5:
        return np.s_[:, :, 0]
    elif fn == 6:
        return np.s_[:, :, -1]


inputFile = sys.argv[1]
bcFam = inputFile.split(".")[0]
# Use the input file name as the alpha directory
outputDir = bcFam + "Alphas"
if not os.path.exists(outputDir):
    raise IOError(f"Cannot find your alpha files in the usual place, {outputDir}.")

###############################################################################
# Read in SEM parameters
###############################################################################
with open(inputFile, "r") as f:
    seminp = yaml.load(f, Loader=yaml.FullLoader)
nframes = seminp["nframes"]
###############################################################################
# PEREGRINE stuff (read in grid and make the patches)
###############################################################################
# Only rank ones reads in the grid
nblks = len(
    [
        f
        for f in os.listdir(seminp["gridPath"])
        if f.startswith("g.") and f.endswith(".h5")
    ]
)
if nblks == 0:
    nblks = len([f for f in os.listdir(seminp["gridPath"]) if f == "grid"])
if nblks == 0:
    raise ValueError(f'Cant find any grid files in {seminp["gridPath"]}')

mb = pg.multiBlock.grid(nblks)
pg.readers.readGrid(mb, seminp["gridPath"])
pg.readers.readConnectivity(mb, seminp["connPath"])
inletBlocks = []
inletFaces = []
for blk in mb:
    for face in blk.faces:
        if face.bcFam == bcFam:
            inletBlocks.append(blk.nblki)
            inletFaces.append(face.nface)
print(f"\nFound {len(inletBlocks)} blocks for this inlet.")

# Flip the grid if it is oriented upside down in the true grid
if seminp["flipdom"]:
    for bn in inletBlocks:
        blk = mb.getBlock(bn)
        blk.array["y"] = -blk.array["y"]
        blk.array["z"] = -blk.array["z"]
# Determinw the extents the grid needs to be shifted
ymin = np.inf
zmin = np.inf
for bn, fn in zip(inletBlocks, inletFaces):
    blk = mb.getBlock(bn)
    face = blk.getFace(fn)
    s1_ = getSlice(fn)
    ymin = min(ymin, blk.array["y"][s1_].min())
    zmin = min(zmin, blk.array["z"][s1_].min())
# Now shift the grid to match the domain (0,0) starting point
for bn in inletBlocks:
    blk = mb.getBlock(bn)
    blk.array["y"] = blk.array["y"] - ymin
    blk.array["z"] = blk.array["z"] - zmin
    # Compute the locations of face centers
    mb.computeMetrics(fdOrder=2)

###############################################################################
# Build the plotting surfaces
###############################################################################

y = np.array([])
z = np.array([])

for bn, fn in zip(inletBlocks, inletFaces):
    blk = mb.getBlock(bn)
    face = blk.getFace(fn)
    s1_ = getSlice(fn)
    y = np.concatenate((y, blk.array["iyc"][s1_].ravel()))
    z = np.concatenate((z, blk.array["izc"][s1_].ravel()))

tri = mpl.tri.Triangulation(z, y)

###############################################################################
# Make plots
###############################################################################
if mpl.checkdep_usetex(True):
    plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 12
plt.rcParams["image.cmap"] = "autumn"

alphasU = np.array([]).reshape(nframes - 1, 0)
alphasV = np.array([]).reshape(nframes - 1, 0)
alphasW = np.array([]).reshape(nframes - 1, 0)
for bn, fn in zip(inletBlocks, inletFaces):
    fileName = outputDir + "/alphas_{}_{}.npy".format(bn, fn)
    temp = np.array([])
    with open(fileName, "rb") as f:
        # The coefficients are in "reverse" order, i.e. the last
        # of the four coefficients is the constant value at the
        # t-t[i] frame location. So we just plot that.
        au = np.load(f)[-1, :, :, :]
        av = np.load(f)[-1, :, :, :]
        aw = np.load(f)[-1, :, :, :]
        shape = au.shape
        au = au.ravel().reshape((nframes - 1, np.prod(au.shape[1::])))
        av = av.ravel().reshape((nframes - 1, np.prod(av.shape[1::])))
        aw = aw.ravel().reshape((nframes - 1, np.prod(aw.shape[1::])))
        alphasU = np.hstack((alphasU, au))
        alphasV = np.hstack((alphasV, av))
        alphasW = np.hstack((alphasW, aw))


def animate(i, Tri, alphas, comp):
    global tcf
    for c in tcf.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    tcf = ax.tricontourf(Tri, alphas[i], levels=levels)
    ax.set_title(f"Frame {i+1}")
    ax.set_aspect("equal")
    return tcf


###############################################################################
# Film u
###############################################################################
plt.rcParams["image.cmap"] = "plasma"
print("Animating U")
fig, ax = plt.subplots()
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$y$")
ax.set_aspect("equal")

levels = np.linspace(0, alphasU.max(), 100)
ticks = np.linspace(0, alphasU.max(), 11)
tcf = ax.tricontourf(tri, np.zeros(alphasU[0].shape), levels=levels)
ratio = y.max() / z.max()
cb = plt.colorbar(
    tcf,
    ticks=ticks,
    fraction=0.046 * ratio,
    pad=0.05,
    label=r"$\overline{U}+u^{\prime}$",
)


anim = FuncAnimation(
    fig,
    animate,
    frames=seminp["nframes"] - 1,
    fargs=(tri, alphasU, "u"),
    repeat=False,
)
try:
    anim.save("U.mp4", writer=mpl.animation.FFMpegWriter(fps=10))
except FileNotFoundError:
    anim.save("U.gif", writer=mpl.animation.PillowWriter(fps=10))

plt.cla()

###############################################################################
# Film v
###############################################################################
plt.rcParams["image.cmap"] = "seismic"
print("Animating v")

fig, ax = plt.subplots()
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$y$")
ax.set_aspect("equal")

symm = max(alphasV.max(), abs(alphasV.min()))
levels = np.linspace(-symm, symm, 100)
ticks = np.linspace(-symm, symm, 11)
tcf = ax.tricontourf(tri, np.zeros(alphasV[0].shape), levels=levels)
cb = plt.colorbar(
    tcf,
    ticks=ticks,
    fraction=0.046 * ratio,
    pad=0.05,
    label=r"$v^{\prime}$",
)

anim = FuncAnimation(
    fig,
    animate,
    frames=seminp["nframes"] - 1,
    fargs=(tri, alphasV, "v"),
    repeat=False,
)
try:
    anim.save("v.mp4", writer=mpl.animation.FFMpegWriter(fps=10))
except FileNotFoundError:
    anim.save("v.gif", writer=mpl.animation.PillowWriter(fps=10))

plt.cla()

###############################################################################
# Film w
###############################################################################
print("Animating w")

fig, ax = plt.subplots()
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$y$")
ax.set_aspect("equal")

symm = max(alphasW.max(), abs(alphasW.min()))
symm = float(f"{symm:.2e}")
levels = np.linspace(-symm, symm, 100)
ticks = np.linspace(-symm, symm, 11)
tcf = ax.tricontourf(tri, np.zeros(alphasW[0].shape), levels=levels)
cb = plt.colorbar(
    tcf,
    ticks=ticks,
    fraction=0.046 * ratio,
    pad=0.05,
    label=r"$w^{\prime}$",
)

anim = mpl.animation.FuncAnimation(
    fig,
    animate,
    frames=seminp["nframes"] - 1,
    fargs=(
        tri,
        alphasW,
        "w",
    ),
    repeat=False,
)
try:
    anim.save("w.mp4", writer=mpl.animation.FFMpegWriter(fps=10))
except FileNotFoundError:
    anim.save("w.gif", writer=mpl.animation.PillowWriter(fps=10))

plt.cla()
