""" helper_functions.py

module implementing various helper functions for exercise 04

Supplementary material for the lecture "Computational Photonics" held at
Leibniz University Hannover in summer term 2017

AUTHOR: OM
DATE: 2020-06-01
"""
import numpy as np


def  energy(t,A):
    """Pulse energy

    Function calculating the energy for the time-domain pulse envelope. This
    can be used to monitor energy conservation for the nonlinear Schroedinger
    equation [1].

    Args:
        t (array): time axis
        A (array): time domain pulse profile

    Returns:
        E (float): energy of the pulse envelope

    Refs:
        [1] Split-Step Methods for the Solution of the Nonlinear
            Schroedinger Equation
            J.A.C. Weideman and B.M. Herbst
            SIAM J. Math. Num. Analysis, 23 (1986) 485
    """
    return np.trapz(np.abs(A)**2,x=t)


def dispersionLength(t0,beta2):
    """Dispersion length

    Args:
        t0 (float): pulse duration
        beta2 (float): 2nd order dispersion parameter

    Returns:
        LD (float):  dispersion length
    """
    return t0*t0/np.abs(beta2)


def nonlinearLength(gamma,APeak):
    """Nonlinear length

    Args:
        gamma (float): nonlinear parameter
        APeak (float): peak amplitude

    Returns:
        LNL (float): nonlinear length
    """
    return 1./gamma/APeak/APeak


def solitonOrder(t0,APeak,beta2,gamma):
   """Soliton Order

    Args:
        t0 (float): pulse duration
        APeak (float): peak amplitude
        beta2 (float): 2nd order dispersion parameter
        gamma (float): nonlinear parameter

    Returns:
        N (float): soliton order
   """
   return np.sqrt(dispersionLength(t0,beta2)/nonlinearLength(gamma,APeak))


# EOF: helper_functions.py
