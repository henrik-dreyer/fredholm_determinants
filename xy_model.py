'''
This file contains helper functions specific to the
XY-model, i.e., the ground state coherent function
its time evolution, cf. Etienne's and Fabian's notes
'''


import numpy as np


def theta(h,gamma,k):
    out = np.arctan( (h-np.cos(k)) / gamma / np.sin(k) )
    return out

def K(hbar, gammabar, h, gamma, k):
    out = np.tan( (theta(hbar,gammabar,k) - theta(h,gamma,k)) /2 )
    return out

def dfdt(f, k, gamma, h):

    eps = np.sqrt( (h-np.cos(k))**2 + gamma**2 * np.sin(k)**2 )
    cosdelta = 2 * ( np.cos(k)**2 + gamma**2 * np.sin(k)**2 - h*np.cos(k) ) / eps
    sindelta = 2 * (-h * np.sin(k) + (1-gamma) * np.sin(k)*np.cos(k) ) / eps

    out = eps * (1 + f**2) * sindelta - 2*1j*eps*f*cosdelta
    return out