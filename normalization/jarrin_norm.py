import numpy as np


def jarrin_norm(signal, domain):
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

    if domain.flow_type in ["channel", "free_shear"]:
        VB = (
            (2 * domain.sigma_x_max)
            * (domain.y_height + 2 * domain.sigma_y_max)
            * (domain.z_width + 2 * domain.sigma_z_max)
        )
        Vdom = (
            (domain.x_length + 2 * domain.sigma_x_max)
            * (domain.y_height + 2 * domain.sigma_y_max)
            * (domain.z_width + 2 * domain.sigma_z_max)
        )
    elif domain.flow_type == "bl":
        VB = (
            (2 * domain.sigma_x_max)
            * (domain.delta + 2 * domain.sigma_y_max)
            * (domain.z_width + 2 * domain.sigma_z_max)
        )
        Vdom = (
            (domain.x_length + 2 * domain.sigma_x_max)
            * (domain.delta + 2 * domain.sigma_y_max)
            * (domain.z_width + 2 * domain.sigma_z_max)
        )

    # We compute the "Eddy Volume" and "Neddy" from the original SEM
    Neddy = domain.neddy * VB / Vdom

    signal = np.sqrt(VB) / np.sqrt(Neddy) * signal

    return signal
