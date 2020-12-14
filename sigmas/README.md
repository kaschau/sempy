**SIGMAS**

This module contains the means of providing interpolation functions for SEM
eddy sizes (a.k.a sigmas, see Jarrin thesis) of various flow types.

The goal is to create a interpolation routine that one can pass a y=height above the wall
value to, and return a [3x3] numpy array of the sigma array for u,v,w velocity fluctuation
components. See Jarrin thesis if you have no idea what that means.

The name of this function MUST be *add_sigmas()*

These profiles are meant to be specific for a specific case, as in, no non-dimensionalization.
It takes in a constructed domiain, with set flow properties (Ublk, utau, viscosity, 
delta, etc.) and builds an interpolation routine for that specific flow feature.
