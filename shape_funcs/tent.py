import numpy as np

def tent(x_dist,y_dist,z_dist,x_sigma,y_sigma,z_sigma):

    #Individual f(x) 'tent' functions for each contributing point
    fxx = np.sqrt(1.5)*(1.0-np.abs(x_dist.reshape((x_dist.shape[0],1)))/x_sigma)
    fxy = np.sqrt(1.5)*(1.0-np.abs(y_dist.reshape((y_dist.shape[0],1)))/y_sigma)
    fxz = np.sqrt(1.5)*(1.0-np.abs(z_dist.reshape((z_dist.shape[0],1)))/z_sigma)

    np.clip(fxx,0.0,None,out=fxx)
    np.clip(fxy,0.0,None,out=fxy)
    np.clip(fxz,0.0,None,out=fxz)

    #Total f(x) from each contributing point
    return fxx*fxy*fxz
