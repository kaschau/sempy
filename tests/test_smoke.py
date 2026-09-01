"""Smoke tests for sempy.

Builds a small Moser channel domain and checks that generated fluctuations
have the right shapes and approximately reproduce the target statistics.

Run with pytest, or directly: python tests/test_smoke.py
"""

import numpy as np

import sempy


def buildChannel():
    reTau = 587.19
    delta = 0.05
    viscosity = 1.81e-5
    utau = reTau * viscosity / delta
    Uo = 2.1263e01 * utau
    tme = 10 * 2 * np.pi * delta / Uo
    yHeight = 2.0 * delta
    zWidth = np.pi * delta

    domain = sempy.geometries.box(
        "channel", Uo, tme, yHeight, zWidth, delta, utau, viscosity
    )
    domain.setSemData(sigmasFrom="jarrin", statsFrom="moser", profileFrom="channel")
    domain.populate(1.0, method="random")
    domain.generateEps()
    domain.computeSigmas()
    domain.makePeriodic(periodicZ=True)
    return domain


def test_domainSlots():
    # domain carries utau and randseed through __slots__
    d = sempy.domain.domain(1.0, 1.0, 1.0, 0.05, 1e-5)
    d.utau = 0.05
    d.randseed = np.random.RandomState(1)
    assert d.yp1 == 1e-5 / 0.05


def test_channelPrimes():
    domain = buildChannel()
    assert domain.neddy > 0

    nframes = 60
    ys = np.linspace(0.05 * domain.delta, 1.95 * domain.delta, 6)
    zs = np.ones(ys.shape[0]) * domain.zWidth / 2.0

    up, vp, wp = sempy.generatePrimes(
        ys, zs, domain, nframes, normalization="exact", progress=False
    )

    for p in (up, vp, wp):
        assert p.shape == (nframes, ys.shape[0])
        assert np.all(np.isfinite(p))

    # exact normalization imposes the Cholesky of Rij on a zero-mean,
    # unit-variance signal, so diagonal stresses should land close to target
    Rij = domain.rijInterp(ys)
    uu = np.mean(up**2, axis=0)
    assert np.allclose(uu, Rij[:, 0, 0], rtol=0.35)
    assert np.allclose(np.abs(np.mean(up, axis=0)), 0.0, atol=1e-8)


def test_blobShape():
    domain = buildChannel()
    ys = np.array([domain.delta])
    zs = np.array([domain.zWidth / 2.0])
    up, vp, wp = sempy.generatePrimes(
        ys, zs, domain, 20, normalization="exact", shape="blob", progress=False
    )
    assert np.all(np.isfinite(up))


def test_badKeywords():
    domain = buildChannel()
    ys = np.array([domain.delta])
    zs = np.array([domain.zWidth / 2.0])
    for kwargs in (
        {"normalization": "bogus"},
        {"normalization": "exact", "shape": "bogus"},
        {"normalization": "exact", "convect": "bogus"},
    ):
        try:
            sempy.generatePrimes(ys, zs, domain, 20, progress=False, **kwargs)
        except NameError:
            pass
        else:
            raise AssertionError(f"expected NameError for {kwargs}")


if __name__ == "__main__":
    test_domainSlots()
    test_channelPrimes()
    test_blobShape()
    test_badKeywords()
    print("All smoke tests passed.")
