import numpy as np


def jarrinNorm(signal, domain):
    """
    Original normalization from Jarrin's thesis

    Parameters:
    -----------
      signal   : numpy.array
            Array of shape(N,3) where N is the length of the signal
      sigmas : sempy.domain
            This normalization needs a lot of information about the domain, so we just pass
            the whole thing in
    Returns:
    --------
        signal : numpy.arrray
            Array of shape(N,3) of the fluctuation components normalized
    """

    if domain.flowType in ["channel", "free_shear"]:
        VB = (
            (2 * domain.sigmaXMax)
            * (domain.yHeight + 2 * domain.sigmaYMax)
            * (domain.zWidth + 2 * domain.sigmaZMax)
        )
        Vdom = (
            (domain.xLength + 2 * domain.sigmaXMax)
            * (domain.yHeight + 2 * domain.sigmaYMax)
            * (domain.zWidth + 2 * domain.sigmaZMax)
        )
    elif domain.flowType == "bl":
        VB = (
            (2 * domain.sigmaXMax)
            * (domain.delta + 2 * domain.sigmaYMax)
            * (domain.zWidth + 2 * domain.sigmaZMax)
        )
        Vdom = (
            (domain.xLength + 2 * domain.sigmaXMax)
            * (domain.delta + 2 * domain.sigmaYMax)
            * (domain.zWidth + 2 * domain.sigmaZMax)
        )

    # We compute the "Eddy Volume" and "Neddy" from the original SEM
    Neddy = domain.neddy * VB / Vdom

    signal = np.sqrt(VB) / np.sqrt(Neddy) * signal

    return signal
