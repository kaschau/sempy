import numpy as np


def exact_norm(signal):
    """
    Exact normalization, this normalization explicitely creates a signal of zero mean, unit variance.
    We no longer enforce zero covariance, a it was generating bad signals. See issue #8.

    Parameters:
    -----------
      signal   : numpy.array
            Array of shape(N,3) where N is the length of the signal
    Returns:
    --------
        signal : numpy.arrray
            Array of shape(N,3) of the fluctuation components normalized
    """

    # Zero out mean
    signal = signal - np.mean(signal, axis=0)

    ################################################################
    # Here lies the whitening procedure, not meant for this world...
    ################################################################
    ##whiten data to eliminate random covariance
    # cov = np.cov(signal,rowvar=False,bias=True)
    # eig_vec = np.linalg.eig(cov)[1]
    ##order eigenvectors to minimize the effect on the original signals
    # order = []
    # perm = np.copy(eig_vec)
    # for row in perm:
    #    maxi = np.argmax(np.abs(row))
    #    while maxi in order:
    #        row[maxi] = 0.0
    #        maxi = np.argmax(np.abs(row))
    #    order.append(maxi)
    # eig_vec = eig_vec[:,order]
    ##rescale any eigenvectors with negative diagonals to be positive
    # if eig_vec[0,0] < 0:
    #    eig_vec[:,0] *= -1
    # if eig_vec[1,1] < 0:
    #    eig_vec[:,1] *= -1
    # if eig_vec[2,2] < 0:
    #    eig_vec[:,2] *= -1
    # signal = np.matmul(eig_vec.T,signal.T).T
    ################################################################
    # RIP
    ################################################################

    # Set variance of each signal to 1
    norm_factor = np.sqrt(np.mean(signal ** 2, axis=0))
    signal = signal / norm_factor

    return signal
