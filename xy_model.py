'''
This file contains helper functions specific to the
XY-model, i.e., the ground state coherent function
its time evolution, cf. Etienne's and Fabian's notes
'''


import numpy as np


def theta(h,gamma,k):
  if h=='inf':
    out = np.zeros(len(p))
  else:
    out = np.arctan( (h-np.cos(k)) / gamma / np.sin(k) )
  return out


def K(hbar, gammabar, h, gamma, k):
    out = np.tan( (theta(hbar,gammabar,k) - theta(h,gamma,k)) /2 )
    return out

def dfdt(f, k, gamma, h):

    eps = np.sqrt( (h-np.cos(k))**2 + gamma**2 * np.sin(k)**2 )
    cosdelta = 2 * ( np.cos(k)**2 + gamma**2 * np.sin(k)**2 - h*np.cos(k) )
    sindelta = 2 * (-h * np.sin(k) + (1-gamma) * np.sin(k)*np.cos(k) )

    try:
    out =  (1 + f**2) * sindelta - 2*1j*f*cosdelta
    except RuntimeWarning
    return out



def h_func(k, h, gamma):
    """
    TODO: eq(51) in xy_field.pdf
    :returns: Coherent State with same R and Q as f
    """
    pass

def rho_s(hk):
    """
    TODO: eq(61) in xy_field.pdf
    """
    pass