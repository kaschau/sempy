

def jarrin_norm(signal,domain):

    if domain.flow_type in ['channel','free_shear']:
        VB = (2*domain.sigma_x_max)*(domain.y_height+2*domain.sigma_y_max)*(domain.z_width+2*domain.sigma_z_max)
        Vdom =(domain.x_length+2*domain.sigma_x_max)*(domain.y_height+2*domain.sigma_y_max)*(domain.z_width+2*domain.sigma_z_max)
    elif domain.flow_type == 'bl':
        VB = (2*domain.sigma_x_max)*(domain.delta+2*domain.sigma_y_max)*(domain.z_width+2*domain.sigma_z_max)
        Vdom =(domain.x_length+2*domain.sigma_x_max)*(domain.delta+2*domain.sigma_y_max)*(domain.z_width+2*domain.sigma_z_max)

    #We compute the "Eddy Volume" and "Neddy" from the original SEM
    Neddy = domain.neddy * VB/Vdom

    return np.sqrt(VB)/np.sqrt(Neddy) * signal
