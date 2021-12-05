#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This utility plots the results of sem4raptor by plotting the U,v,w
components at the values of the frames, and creates animation
videos of the results.

Uses identical input file that was used in sem4raptor. Expects the alphas to be located
in ./<sem4raptor.inp>_alphas

Creats three videos, U.mp4, v.mp4, and w.mp4

Plots are all non dimensional raptor values.

Example
-------
/path/to/sempy/utilities/animate_alphas.py <sem.inp>

"""

import raptorpy as rp
import numpy as np
from scipy.io import FortranFile
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

input_file = sys.argv[1]

# Use the input file name as the alpha directory
output_dir = input_file.split(".")[0] + "_alphas"
if not os.path.exists(output_dir):
    raise IOError(f"Cannot find your alpha files in the usual place, {output_dir}.")

############################################################################################
# Read in SEM parameters
############################################################################################
# The zeroth rank will read in the input file and give it to the other ranks
seminp = dict()
with open(input_file, "r") as f:
    for line in [
        i
        for i in f.readlines()
        if not i.replace(" ", "").startswith("#") and i.strip() != ""
    ]:
        nocomment = line.strip().split("#")[0]
        key, val = tuple(nocomment.replace(" ", "").split("="))
        try:  # convert numbers to floats or ints
            if "." in val:
                seminp[key] = float(val)
            else:
                seminp[key] = int(val)
        except ValueError:
            if val in ["True", "true", "t", "T"]:
                seminp[key] = True
            elif val in ["False", "true", "f", "F"]:
                seminp[key] = False
            else:
                seminp[key] = val

############################################################################################
# Read in patches.txt
############################################################################################
# Only the zeroth rank reads in patches.txt
patchNum = []
blockNum = []
with open("patches.txt", "r") as f:
    for line in [i for i in f.readlines() if not i.startswith("#")]:
        li = line.strip().split(",")
        patchNum.append(int(li[0]))
        blockNum.append(int(li[1]))

npatches = len(patchNum)

############################################################################################
# RAPTOR stuff (read in grid and make the patches)
############################################################################################
# Only rank ones reads in the grid
nblks = len([f for f in os.listdir(seminp["grid_path"]) if f.startswith("g.")])
if nblks == 0:
    nblks = len([f for f in os.listdir(seminp["grid_path"]) if f == "grid"])
if nblks == 0:
    raise ValueError(f'Cant find any grid files in {seminp["grid_path"]}')

mb = rp.multiblock.grid(nblks)
rp.readers.read_raptor_grid(mb, seminp["grid_path"])
# Flip the grid if it is oriented upside down in the true grid
if seminp["flipdom"]:
    for bn in blockNum:
        blk = mb[bn - 1]
        blk.y = -blk.y
        blk.z = -blk.z
# Determinw the extents the grid needs to be shifted
ymin = np.inf
zmin = np.inf
for bn in blockNum:
    blk = mb[bn - 1]
    ymin = min(ymin, blk.y[:, :, 0].min())
    zmin = min(zmin, blk.z[:, :, 0].min())
# Now shift the grid to match the domain (0,0) starting point
for bn in blockNum:
    blk = mb[bn - 1]
    blk.y = blk.y - ymin
    blk.z = blk.z - zmin
    # Compute the locations of face centers
    blk.compute_U_face_centers()
    blk.compute_V_face_centers()
    blk.compute_W_face_centers()

############################################################################################
# Build the plotting surfaces
############################################################################################

uy = np.array([])
uz = np.array([])
vy = np.array([])
vz = np.array([])
wy = np.array([])
wz = np.array([])

ny = []
nz = []
for bn in blockNum:
    blk = mb[bn - 1]
    uy = np.concatenate((uy, blk.yu[:, :, 0].ravel()))
    uz = np.concatenate((uz, blk.zu[:, :, 0].ravel()))
    vy = np.concatenate((vy, blk.yv[:, :, 0].ravel()))
    vz = np.concatenate((vz, blk.zv[:, :, 0].ravel()))
    wy = np.concatenate((wy, blk.yw[:, :, 0].ravel()))
    wz = np.concatenate((wz, blk.zw[:, :, 0].ravel()))
    ny.append(blk.ny)
    nz.append(blk.nz)

TriU = mpl.tri.Triangulation(uz, uy)
TriV = mpl.tri.Triangulation(vz, vy)
TriW = mpl.tri.Triangulation(wz, wy)

############################################################################################
# Function to read in frames
############################################################################################


def getFrame(frame, comp):
    alphas = np.array([])
    for i, pn in enumerate(patchNum):
        file_name = output_dir + "/alphas_{:06d}_{:04d}".format(pn, frame)
        with FortranFile(file_name, "r") as f90:
            if comp == "u":
                a = f90.read_reals(dtype=np.float64).reshape(4, nz[i] - 1, ny[i] - 1)[
                    -1, :, :
                ]
            elif comp == "v":
                _ = f90.read_reals(dtype=np.float64)
                a = f90.read_reals(dtype=np.float64).reshape(4, nz[i] - 1, ny[i])[
                    -1, :, :
                ]
            elif comp == "w":
                _ = f90.read_reals(dtype=np.float64)
                _ = f90.read_reals(dtype=np.float64)
                a = f90.read_reals(dtype=np.float64).reshape(4, nz[i], ny[i] - 1)[
                    -1, :, :
                ]
            else:
                raise ValueError(f"Unknown component {comp}")

            alphas = np.concatenate((alphas, a.ravel()))

    return alphas


############################################################################################
# Make plots
############################################################################################
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


############################################################################################
# Film u
############################################################################################
print("Animating U")
fig, ax = plt.subplots()
ax.set_xlabel(r"$z/L_{ref}$")
ax.set_ylabel(r"$y/L_{ref}$")
ax.set_aspect("equal")

alphas = getFrame(1, "u")
levels = np.linspace(0, alphas.max(), 100)
ticks = np.linspace(0, alphas.max(), 11)
tcf = ax.tricontourf(TriU, np.zeros(alphas.shape), levels=levels)
ratio = uy.max() / uz.max()
cb = plt.colorbar(
    tcf,
    ticks=ticks,
    fraction=0.046 * ratio,
    pad=0.05,
    label=r"$\overline{U}+u^{\prime}$",
)


anim = mpl.animation.FuncAnimation(
    fig, animate, frames=seminp["nframes"] - 1, fargs=(TriU, "u"), repeat=False
)
try:
    anim.save("U.mp4", writer=mpl.animation.FFMpegWriter(fps=10))
except FileNotFoundError:
    anim.save("U.gif", writer=mpl.animation.PillowWriter(fps=10))

plt.cla()

############################################################################################
# Film v
############################################################################################
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
tcf = ax.tricontourf(TriV, alphas, levels=levels)
ratio = vy.max() / vz.max()
cb = plt.colorbar(
    tcf, ticks=ticks, fraction=0.046 * ratio, pad=0.05, label=r"$v^{\prime}$"
)

anim = mpl.animation.FuncAnimation(
    fig,
    animate,
    frames=seminp["nframes"] - 1,
    fargs=(
        TriV,
        "v",
    ),
    repeat=False,
)
try:
    anim.save("v.mp4", writer=mpl.animation.FFMpegWriter(fps=10))
except FileNotFoundError:
    anim.save("v.gif", writer=mpl.animation.PillowWriter(fps=10))

plt.cla()

############################################################################################
# Film w
############################################################################################
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
tcf = ax.tricontourf(TriW, np.zeros(alphas.shape), levels=levels)
ratio = wy.max() / wz.max()
cb = plt.colorbar(
    tcf, ticks=ticks, fraction=0.046 * ratio, pad=0.05, label=r"$w^{\prime}$"
)

anim = mpl.animation.FuncAnimation(
    fig,
    animate,
    frames=seminp["nframes"] - 1,
    fargs=(
        TriW,
        "w",
    ),
    repeat=False,
)
try:
    anim.save("w.mp4", writer=mpl.animation.FFMpegWriter(fps=10))
except FileNotFoundError:
    anim.save("w.gif", writer=mpl.animation.PillowWriter(fps=10))

plt.cla()
