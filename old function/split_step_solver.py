""" split_step_solver.py

module implementing different versions of the split step Fourier method
for the solution of the nonlinear Schroedinger equation (NSE; part 1)
and the higher-order nonlinear Schroedinger equation (HONSE; part 2)

Supplementary material for the lecture "Computational Photonics" held at
Leibniz University Hannover in summer term 2017

AUTHOR: OM
DATE: 2020-06-01
"""
import numpy as np
import numpy.fft as nfft

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft


def SSFM_NSE_simple(z, t, A0_t, beta2, gamma, nSkip):
    """Split step fourier method using simple operator splitting

    Implements divid-and-conquer strategy to solve the nonlinear Schroedinger
    equation (NLS). Pulse propagation is performed using a simple splitting
    scheme.

    NOTES:
        - uses abbreviations FT, specifying the DFT, and IFT, specifying its
          inverse. These are defined at the beginning of the script right
          beneath the import statements.

    Args:
        z (array): samples along propagation distance
        t (array): time samples
        A0_t (array): time domain field envelope
        beta2 (float): 2nd order dispersion parameter
        gamma (float): nonlinear parameter
        nSkip (int): keep only each nSkip-th field configuration

    Returns: (z,Azt)
        z (array): resulting z-samples at which field envelope is recorded
        Azt (array): resulting time domain field envelope
    """
    dz = z[1]-z[0]
    dt = t[1]-t[0]
    A_t  = np.copy(A0_t)
    w = nfft.fftfreq(t.size,d=dt)*2*np.pi

    # -- INITIALIZE DATA STRUCTURES THAT WILL ACCUMLATE RESULTS
    res_z = []; res_z.append(0)
    res_A = []; res_A.append(A0_t)

    for idx in range(1,z.size):
        A_t =  A_t*np.exp(1j*gamma*np.abs(A_t)**2*dz)   # nonlinear sub-step
        A_t =  IFT(np.exp(1j*0.5*beta2*w*w*dz)*FT(A_t)) # linear sub-step

        # -- KEEP ONLY EVERY nSkip-TH FIELD CONFIGURATION 
        if idx%nSkip==0:
            res_z.append(z[idx]) # keep z-value
            res_A.append(A_t)    # keep field

    return np.asarray(res_z), np.asarray(res_A)


def SSFM_HONSE_symmetric(z, t, A0_t, beta2, beta3, beta4, gamma, nSkip):
    """Split step fourier method using symmetric operator splitting

    Implements divid-and-conquer strategy to solve the nonlinear Schroedinger
    equation (NLS). Pulse propagation is performed using a symmetric splitting
    scheme.

    NOTES:
        - uses abbreviations FT, specifying the DFT, and IFT, specifying its
          inverse. These are defined at the beginning of the script right
          beneath the import statements.

    Args:
        z (array): samples along propagation distance
        t (array): time samples
        A0_t (array): time domain field envelope
        beta2 (float): 2nd order dispersion parameter
        gamma (float): nonlinear parameter
        nSkip (int): keep only each nSkip-th field configuration

    Returns: (z,Azt)
        z (array): resulting z-samples at which field envelope is recorded
        Azt (array): resulting time domain field envelope
    """
    dz = z[1]-z[0]
    dt = t[1]-t[0]
    A_t  = np.copy(A0_t)
    w = nfft.fftfreq(t.size,d=dt)*2*np.pi

    # -- INITIALIZE DATA STRUCTURES THAT WILL ACCUMLATE RESULTS
    res_z = []; res_z.append(0)
    res_A = []; res_A.append(A0_t)

    for idx in range(1,z.size):
        # TODO: THE LINES BELOW RE-IMPLEMENT THE SPLITTING SCHEME USED BY THE
        # FUNCTION splitStepFourierMethod_NSE_simple(). MODIFY IT SO THAT
        # IT REALIZES THE SYMMETRIC SPLITTING SCHEME
        A_t =  IFT(np.exp(1j*beta2*w*w*dz/2/2 + 1j*beta3*w**3*dz/2/6 + 1j*beta4*w**4*dz/2/24)*FT(A_t)) 
        A_t =  A_t*np.exp(1j*gamma*np.abs(A_t)**2*dz)       # full nonlinear sub-step
        A_t =  IFT(np.exp(1j*beta2*w*w*dz/2/2 + 1j*beta3*w**3*dz/2/6 + 1j*beta4*w**4*dz/2/24)*FT(A_t)) 

        # -- KEEP ONLY EVERY nSkip-TH FIELD CONFIGURATION 
        if idx%nSkip==0:
            res_z.append(z[idx]) # keep z-value
            res_A.append(A_t)    # keep field

    return np.asarray(res_z), np.asarray(res_A)


# EOF: split_step_solver.py
