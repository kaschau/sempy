import numpy as np
import scipy.integrate as integrate
from sempy.domain import domain


class box(domain):
    def __init__(self, flowType, Uo, tme, yHeight, zWidth, delta, utau, viscosity):

        super().__init__(Uo, tme, delta, utau, viscosity)
        self.yHeight = yHeight
        self.zWidth = zWidth
        self.flowType = flowType
        self.randseed = np.random.RandomState(10101)

    def populate(self, cEddy=1.0, method="random"):

        if self.sigmaInterp is None:
            raise ValueError(
                "Please set your flow data before trying to populate your domain"
            )

        self.eddyPopMethod = method

        # generate eddy volume
        if method == "random":
            # Set the extents of the eddy volume in x,y,z
            lows = [
                0.0 - self.sigmaXMax,
                0.0 - self.sigmaYMax,
                0.0 - self.sigmaZMax,
            ]
            if self.flowType in ["channel", "freeshear"]:
                highs = [
                    self.xLength + self.sigmaXMax,
                    self.yHeight + self.sigmaYMax,
                    self.zWidth + self.sigmaZMax,
                ]
            elif self.flowType == "bl":
                highs = [
                    self.xLength + self.sigmaXMax,
                    self.delta + self.sigmaYMax,
                    self.zWidth + self.sigmaZMax,
                ]
            # Compute number of eddys
            VB = np.product(np.array(highs) - np.array(lows))
            neddy = int(cEddy * VB / self.vSigmaMin)
            # Generate random eddy locations
            self.eddyLocs = self.randseed.uniform(low=lows, high=highs, size=(neddy, 3))

        elif method == "PDF":
            # Set the extents of the eddy volume in x,y,z
            lows = [
                0.0 - self.sigmaXMax,
                0.0 - np.max(self.sigmaInterp(0.0)[:, 1]),
                0.0 - self.sigmaZMax,
            ]
            if self.flowType in ["channel", "freeshear"]:
                highs = [
                    self.xLength + self.sigmaXMax,
                    self.yHeight + np.max(self.sigmaInterp(self.yHeight)[:, 1]),
                    self.zWidth + self.sigmaZMax,
                ]
            elif self.flowType == "bl":
                highs = [
                    self.xLength + self.sigmaXMax,
                    self.delta + np.max(self.sigmaInterp(self.delta)[:, 1]),
                    self.zWidth + self.sigmaZMax,
                ]
            # Eddy heights as a function of y
            testYs = np.linspace(lows[1], highs[1], 200)
            testSigmas = self.sigmaInterp(testYs)
            # Smallest eddy volume
            vEddy = np.min(np.product(testSigmas, axis=1), axis=1)
            # Find the smallest eddy V and largest eddy V
            vEddyMin = vEddy.min()
            vEddyMax = vEddy.max()
            # This ratio sets how much more likely the small eddy placement is compared to the large eddy placement
            vRatio = vEddyMax / vEddyMin
            # Flip and shift so largest eddy sits on zero
            vEddyPrime = -(vEddy - vEddyMin) + (vEddyMax - vEddyMin)

            # Rescale so lowest point is equal to one
            vEddyPrime = (vEddyPrime / vEddyMax) * vRatio + 1.0
            vEddyNorm = integrate.trapz(vEddyPrime, testYs)
            # Create a PDF of eddy placement in y by normalizing pdf integral
            pdf = vEddyPrime / vEddyNorm
            # Compute average eddy volume
            expectedVeddy = integrate.trapz(pdf * vEddy, testYs)
            # Compute neddy
            VB = np.product(np.array(highs) - np.array(lows))
            neddy = int(cEddy * VB / expectedVeddy)
            # Create bins for placing eddys in y
            dy = np.abs(testYs[1] - testYs[0])
            binCenters = 0.5 * (testYs[1::] + testYs[0:-1])
            binPdf = 0.5 * (pdf[1::] + pdf[0:-1]) * dy
            # place neddys in bins according to pdf
            binLoc = self.randseed.choice(
                np.arange(binPdf.shape[0]), p=binPdf, size=neddy
            )

            # create y values for eddys
            # can only generate values at bin centers, so we add a random value to randomize y placement within the bin
            eddyYs = np.take(binCenters, binLoc) + self.randseed.uniform(
                low=-dy / 2.0, high=dy / 2.0, size=neddy
            )

            # Generate random eddy locations for their x and z locations
            self.eddyLocs = self.randseed.uniform(low=lows, high=highs, size=(neddy, 3))
            # Replace ys with PDF ys
            self.eddyLocs[:, 1] = eddyYs

        else:
            raise NameError(f"Unknown population method : {method}")

        # Now, we remove all eddies whose volume of influence lies completely outside the geometry.
        tempSigmas = self.sigmaInterp(self.eddyLocs[:, 1])
        keepEddys = np.where(
            (self.eddyLocs[:, 0] + np.max(tempSigmas[:, :, 0], axis=1) > 0.0)
            & (self.eddyLocs[:, 0] - np.max(tempSigmas[:, :, 0], axis=1) < self.xLength)
            & (self.eddyLocs[:, 1] + np.max(tempSigmas[:, :, 1], axis=1) > 0.0)
            & (self.eddyLocs[:, 1] - np.max(tempSigmas[:, :, 1], axis=1) < self.yHeight)
            & (self.eddyLocs[:, 2] + np.max(tempSigmas[:, :, 2], axis=1) > 0.0)
            & (self.eddyLocs[:, 2] - np.max(tempSigmas[:, :, 2], axis=1) < self.zWidth)
        )
        self.eddyLocs = self.eddyLocs[keepEddys]
