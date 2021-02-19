**sempy**

Python implementation of SEM with better stuffs too.

The key to understanding how sempy works is to understand how SEM works. Read the Jarrin thesis [here](./References/Papers/Synthetic-Inflow-Boundary-Conditions-for-the-Numerical-Simulation-of-Turbulence_2008.pdf).

There is one key distinctions in how sempy and most SEM implementations work. Instead of creating a small box around our inlet surface and convecting eddys past the inlet plane, sempy creates a mega box and convects the inlet (or just individual points) thorugh the mega box. This leads to a lot of advantages in performance as well as experimentation.

Here is an example of how we [generate fluctuations](./generate_primes.py) in sempy.

We begin by populating a mega box with eddys. This mega box is wide and tall enough to encompass our inlet plane, and long enough to traverse throug it for as long as we want our signal to be.

![All Eddys](./References/readme/all_eddy.png)

With sempy, you generate an entire signal at a single (y,z) point on the inlet plane. When we know this point, we can draw a "time line" down the domain. Traversing down this line is like traversing forward in time.

![All Eddys Line](./References/readme/all_eddy_line.png)

One way we get performance savings is by filtering out all eddys that do not effect any points on this line. We can find this out using the eddy sigma values. All points whos sigmas do not extend across this line will not contribute to this point's fluctiation signal.

![Line Eddys](./References/readme/line_eddy.png)

With this reduced set, we can much more quickly perform the SEM calculations. Next, we start at time = zero. We can further filter out eddys too far into the future to effect this time. We compute the sum of the contributing eddys for this point in time, then march forward to the next point in time. We repeat this until a signal of the desired length is computed. We then move onto the next point of interest.

![Point Eddys](./References/readme/points.gif)
