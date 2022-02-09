#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This utility plots the results of semForPEREGRINE by plotting the u,v,w
components at the values of the frames, and creates animation
videos of the results.

Uses identical input file that was used in semForPEREGRINE. Expects the alphas to be located
in ./<semForPEREGRINE.inp>_alphas

Creats three videos, u.mp4, v.mp4, and w.mp4

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
from mpl.animation import FuncAnimation

inputFile = sys.argv[1]

# Use the input file name as the alpha directory
outputDir = inputFile.split(".")[0] + "_alphas"
if not os.path.exists(outputDir):
    raise IOError(f"Cannot find your alpha files in the usual place, {outputDir}.")

###############################################################################
# Read in SEM parameters
###############################################################################
with open(inputFile, "r") as f:
    seminp = yaml.load(f, Loader=yaml.FullLoader)

###############################################################################
# Read in patches.txt
###############################################################################
# Only the zeroth rank reads in patches.txt
patchNum = []
blockNum = []
with open("patches.txt", "r") as f:
    for line in [i for i in f.readlines() if not i.startswith("#")]:
        li = line.strip().split(",")
        patchNum.append(int(li[0]))
        blockNum.append(int(li[1]))

npatches = len(patchNum)

###############################################################################
# PEREGRINE stuff (read in grid and make the patches)
###############################################################################
# Only rank ones reads in the grid
nblks = len([f for f in os.listdir(seminp["gridPath"]) if f.startswith("g.")])
if nblks == 0:
    nblks = len([f for f in os.listdir(seminp["gridPath"]) if f == "grid"])
if nblks == 0:
    raise ValueError(f'Cant find any grid files in {seminp["gridPath"]}')

mb = pg.multiBlock.grid(nblks)
pg.readers.readGrid(mb, seminp["gridPath"])
# Flip the grid if it is oriented upside down in the true grid
if seminp["flipdom"]:
    for bn in blockNum:
        blk = mb[bn - 1]
        blk.array["y"] = -blk.array["y"]
        blk.array["z"] = -blk.array["z"]
# Determinw the extents the grid needs to be shifted
ymin = np.inf
zmin = np.inf
for bn in blockNum:
    blk = mb[bn - 1]
    ymin = min(ymin, blk.array["y"][0, :, :].min())
    zmin = min(zmin, blk.array["z"][0, :, :].min())
# Now shift the grid to match the domain (0,0) starting point
for bn in blockNum:
    blk = mb[bn - 1]
    blk.array["y"] = blk.array["y"] - ymin
    blk.array["z"] = blk.array["z"] - zmin
    # Compute the locations of face centers
    mb.computeMetrics(fdOrder=2)

###############################################################################
# Build the plotting surfaces
###############################################################################

y = np.array([])
z = np.array([])

ny = []
nz = []
for bn in blockNum:
    blk = mb[bn - 1]
    y = np.concatenate((y, blk.array["iyu"][0, :, :].ravel()))
    z = np.concatenate((z, blk.array["izu"][0, :, :].ravel()))
    ny.append(blk.nj)
    nz.append(blk.nk)

tri = mpl.tri.Triangulation(z, y)

###############################################################################
# Function to read in frames
###############################################################################


def getFrame(frame, comp):
    alphas = np.array([])
    for i, pn in enumerate(patchNum):
        fileName = outputDir + "/alphas_{:06d}_{:04d}".format(pn, frame)
        with open(fileName, "rb") as f:
            # The coefficients are in "reverse" order, i.e. the last
            # of the four coefficients is the constant value at the
            # t-t[i] frame location. So we just plot that.
            if comp == "u":
                a = np.load(f)[-1, :, :]
            elif comp == "v":
                _ = np.load(f)
                a = np.load(f)[-1, :, :]
            elif comp == "w":
                _ = np.load(f)
                _ = np.load(f)
                a = np.load(f)[-1, :, :]
            else:
                raise ValueError(f"Unknown component {comp}")

            alphas = np.concatenate((alphas, a.ravel()))

    return alphas


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

# animation function


def animate(i, Tri, comp):
    global tcf
    for c in tcf.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    alphas = getFrame(i + 1, comp)
    tcf = ax.tricontourf(Tri, alphas, levels=levels)
    ax.set_title(f"Frame {i+1}")
    return tcf


###############################################################################
# Film u
###############################################################################
print("Animating U")
fig, ax = plt.subplots()
ax.set_xlabel(r"$z/L_{ref}$")
ax.set_ylabel(r"$y/L_{ref}$")
ax.set_aspect("equal")

alphas = getFrame(1, "u")
levels = np.linspace(0, alphas.max(), 100)
ticks = np.linspace(0, alphas.max(), 11)
tcf = ax.tricontourf(tri, np.zeros(alphas.shape), levels=levels)
ratio = y.max() / z.max()
cb = plt.colorbar(
    tcf,
    ticks=ticks,
    fraction=0.046 * ratio,
    pad=0.05,
    label=r"$\overline{U}+u^{\prime}$",
)


anim = FuncAnimation(
    fig, animate, frames=seminp["nframes"] - 1, fargs=(tri, "u"), repeat=False
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
ax.set_xlabel(r"$z/L_{ref}$")
ax.set_ylabel(r"$y/L_{ref}$")
ax.set_aspect("equal")

alphas = getFrame(1, "v")
symm = max(alphas.max(), abs(alphas.min()))
levels = np.linspace(-symm, symm, 100)
ticks = np.linspace(-symm, symm, 11)
cb = plt.colorbar(
    tcf, ticks=ticks, fraction=0.046 * ratio, pad=0.05, label=r"$v^{\prime}$"
)

anim = mpl.animation.FuncAnimation(
    fig,
    animate,
    frames=seminp["nframes"] - 1,
    fargs=(
        tri,
        "v",
    ),
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
ax.set_xlabel(r"$z/L_{ref}$")
ax.set_ylabel(r"$y/L_{ref}$")
ax.set_aspect("equal")

alphas = getFrame(1, "w")
symm = max(alphas.max(), abs(alphas.min()))
symm = float(f"{symm:.2e}")
levels = np.linspace(-symm, symm, 100)
ticks = np.linspace(-symm, symm, 11)
cb = plt.colorbar(
    tcf, ticks=ticks, fraction=0.046 * ratio, pad=0.05, label=r"$w^{\prime}$"
)

anim = mpl.animation.FuncAnimation(
    fig,
    animate,
    frames=seminp["nframes"] - 1,
    fargs=(
        tri,
        "w",
    ),
    repeat=False,
)
try:
    anim.save("w.mp4", writer=mpl.animation.FFMpegWriter(fps=10))
except FileNotFoundError:
    anim.save("w.gif", writer=mpl.animation.PillowWriter(fps=10))

plt.cla()
