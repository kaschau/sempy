import numpy as np


class domain:
    def __init__(self, Ublk, totalTime, delta, utau, viscosity):

        self.xLength = Ublk * totalTime
        self.Ublk = Ublk
        self.delta = delta
        self.utau = utau
        self.viscosity = viscosity
        try:
            self.yp1 = viscosity / utau
        except TypeError:
            self.yp1 = None

        self.flowType = None
        self.yHeight = None
        self.zHeight = None

        self.sigmaInterp = None
        self.sigmaXMin = None
        self.sigmaXMax = None
        self.sigmaYMin = None
        self.sigmaYMax = None
        self.sigmaZMin = None
        self.sigmaZMax = None
        self.vSigmaMin = None
        self.vSigmaMax = None

        self.rijInterp = None
        self.ubarInterp = None

        self.eddyLocs = None
        self.eps = None
        self.sigmas = None

        self.profileFrom = None
        self.sigmasFrom = None
        self.statsFrom = None
        self.eddyPopMethod = None

        self._neddy = None

    @property
    def neddy(self):
        if self._neddy is None:
            return self.eddyLocs.shape[0]
        else:
            return self._neddy

    def setSemData(
        self,
        sigmasFrom="jarrin",
        statsFrom="moser",
        profileFrom="channel",
        scaleFactor=1.0,
    ):

        self.profileFrom = profileFrom
        self.sigmasFrom = sigmasFrom
        self.statsFrom = statsFrom

        # MEAN VELOCITY PROFILE
        if profileFrom == "channel":
            from .profiles.channel import addProfile
        elif profileFrom == "bl":
            from .profiles.bl import addProfile
        elif profileFrom == "uniform":
            from .profiles.uniform import addProfile
        else:
            raise NameError(f"Unknown profile keyword : {profileFrom}")

        # SIGMAS
        if sigmasFrom == "jarrin":
            from .sigmas.jarrinChannel import addSigmas
        elif sigmasFrom == "uniform":
            from .sigmas.uniform import addSigmas
        elif sigmasFrom == "linearBL":
            from .sigmas.linearBL import addSigmas
        else:
            raise NameError(f"Unknown sigmas keyword : {sigmasFrom}")

        # STATS
        if statsFrom == "moser":
            from .stats.moserChannel import addStats
        elif statsFrom == "spalart":
            from .stats.spalartBL import addStats
        elif statsFrom == "isotropic":
            from .stats.isotropic import addStats
        else:
            raise NameError(f"Unknown stats keyword : {statsFrom}")

        addProfile(self)
        addSigmas(self, scaleFactor)
        addStats(self)

    def computeSigmas(self):
        # Compute all eddy sigmas as function of y
        self.sigmas = self.sigmaInterp(self.eddyLocs[:, 1])

    def generateEps(self):
        # generate epsilons
        self.eps = np.where(
            self.randseed.uniform(low=-1, high=1, size=(self.neddy, 3)) <= 0.0,
            -1.0,
            1.0,
        )

    def makePeriodic(self, periodicX=False, periodicY=False, periodicZ=False):
        if True in [periodicX, periodicY, periodicZ]:
            print(
                "Making domain periodic in {}".format("x" if periodicX else "")
                + " {}".format("y" if periodicY else "")
                + " {}".format("z" if periodicZ else "")
            )

        # check if we have epsilons or not
        if self.eps is None:
            raise AttributeError(
                "Please generate your domains epsilons before making it periodic"
            )
        if self.sigmas is None:
            raise AttributeError(
                "Please compute your domain's sigmas before making it periodic"
            )
        if self.sigmaInterp is None:
            raise AttributeError(
                "Please set your domain's sigmaInterp before making it periodic"
            )
        # Make periodic
        if periodicX:
            keepEddys = np.where(
                self.eddyLocs[:, 0] + np.max(self.sigmas[:, :, 0], axis=1)
                < self.xLength
            )
            keepEddyLocs = self.eddyLocs[keepEddys]
            keepEps = self.eps[keepEddys]
            periodicEddys = np.where(
                (self.eddyLocs[:, 0] + np.max(self.sigmas[:, :, 0], axis=1) > 0.0)
                & (self.eddyLocs[:, 0] - np.max(self.sigmas[:, :, 0], axis=1) < 0.0)
            )
            periodicEddyLocs = self.eddyLocs[periodicEddys]
            periodicEddyLocs[:, 0] = periodicEddyLocs[:, 0] + self.xLength
            periodicEps = self.eps[periodicEddys]
            self.eddyLocs = np.concatenate((keepEddyLocs, periodicEddyLocs))
            self.eps = np.concatenate((keepEps, periodicEps))
            # Update the sigma array if it has been created already
            self.sigmas = self.sigmaInterp(self.eddyLocs[:, 1])
        if periodicY:
            keepEddys = np.where(
                self.eddyLocs[:, 1] + np.max(self.sigmas[:, :, 1], axis=1)
                < self.yHeight
            )
            keepEddyLocs = self.eddyLocs[keepEddys]
            keepEps = self.eps[keepEddys]
            periodicEddys = np.where(
                (self.eddyLocs[:, 1] + np.max(self.sigmas[:, :, 1], axis=1) > 0.0)
                & (self.eddyLocs[:, 1] - np.max(self.sigmas[:, :, 1], axis=1) < 0.0)
            )
            periodicEddyLocs = self.eddyLocs[periodicEddys]
            periodicEddyLocs[:, 1] = periodicEddyLocs[:, 1] + self.yHeight
            periodicEps = self.eps[periodicEddys]
            self.eddyLocs = np.concatenate((keepEddyLocs, periodicEddyLocs))
            self.eps = np.concatenate((keepEps, periodicEps))
            # Update the sigma array if it has been created already
            self.sigmas = self.sigmaInterp(self.eddyLocs[:, 1])
        if periodicZ:
            keepEddys = np.where(
                self.eddyLocs[:, 2] + np.max(self.sigmas[:, :, 2], axis=1) < self.zWidth
            )
            keepEddyLocs = self.eddyLocs[keepEddys]
            keepEps = self.eps[keepEddys]
            periodicEddys = np.where(
                (self.eddyLocs[:, 2] + np.max(self.sigmas[:, :, 2], axis=1) > 0.0)
                & (self.eddyLocs[:, 2] - np.max(self.sigmas[:, :, 2], axis=1) < 0.0)
            )
            periodicEddyLocs = self.eddyLocs[periodicEddys]
            periodicEddyLocs[:, 2] = periodicEddyLocs[:, 2] + self.zWidth
            periodicEps = self.eps[periodicEddys]
            self.eddyLocs = np.concatenate((keepEddyLocs, periodicEddyLocs))
            self.eps = np.concatenate((keepEps, periodicEps))
            # Update the sigma array if it has been created already
            self.sigmas = self.sigmaInterp(self.eddyLocs[:, 1])

    def __repr__(self):
        print(f"\nFlow Type: {self.flowType}")
        print(f"Domain Length = {self.xLength} [m] ({self.xLength/self.Ublk} [s])")
        print(f"Domain Height = {self.yHeight} [m]")
        print(f"Domain Width = {self.zWidth} [m]")

        print("Flow Parameters:")
        print(f"    U_bulk = {self.Ublk} [m/s]")
        print(f"    delta = {self.delta} [m]")
        print(f"    u_tau = {self.utau} [m/s]")
        print(f"    viscosity(nu) = {self.viscosity} [m^2/s]")
        print(f"    Y+ one = {self.yp1} [m]")

        print(f"Sigmas from {self.sigmasFrom}")
        print(f"Stats from {self.statsFrom}")
        print(f"Profile from {self.profileFrom}")

        print("Sigmas (min,max) [m]:")
        print(f"    x : {self.sigmaXMin},{self.sigmaXMax}")
        print(f"    y : {self.sigmaYMin},{self.sigmaYMax}")
        print(f"    z : {self.sigmaZMin},{self.sigmaZMax}")
        print(f"  Vol : {self.vSigmaMin},{self.vSigmaMax}")

        print(f"Eddy population method : {self.eddyPopMethod}")
        print(f"Number of Eddys: {self.neddy}")
