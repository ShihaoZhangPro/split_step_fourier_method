"""main_ex04_part1_ID.py

Computer Exercises for Lecture "Computational Photonics"

Exercise 04 -  The nonlinear Schroedinger equation - Part I

NOTE: Before you start working on the assignments and before you hand in your
solution, carefully read the instructions below.

    * This assignment is due until 10:30 am. Make sure to hand in the completed
      script scriptName ID.py into the assignment folder Exercise04_solution.
      Assignments handed in after 10:30 am will not be condsidered.

    * You can work on these assignments in groups of up to 3 students. Note,
      however, that everybody has to hand in his/her own solution.

    * Please personalize your completed script main_ex04_part1 ID.py by replacing
      the handle ID by your student-id. For redundancy, make sure to also
      include your name and student-id in the completed scripts that you hand
      in. We therefore provided three metadata variables
          __STUDENT_NAME__ = "YOUR NAME HERE"
          __STUDENT_ID__   = "YOUR STUDENT ID HERE"
          __GROUP_MEMBERS__ = "IF YOU WORKED AS PART OF A GROUP, LIST YOUR GROUP MEMBERS HERE"
      at the beginning of the script.

    * To complete the code, look out for TODO statements, they will provide
      further hints!

DATE: 2021-06-01
"""
import numpy as np
from split_step_solver import SSFM_NSE_simple, SSFM_NSE_symmetric
from figures import figure_1a, figure_1b, figure_1c
from helper_functions import energy, solitonOrder

# -- TODO: PERSONALIZE THIS SCRIPT BY AMENDING THE BELOW METADATA VARIABLES
__STUDENT_NAME__  = "YOUR NAME HERE"
__STUDENT_ID__    = "YOUR_STUDENT_ID_HERE"
__GROUP_MEMBERS__ = "IF YOU WORKED AS PART OF A GROUP, LIST YOUR GROUP MEMBERS HERE"


def main_a():

    # -- SET FIBER PARAMETERS
    beta2 = -0.01276                        # (ps^2/m)
    gamma = 0.045                           # (1/W/m) nonlinear coefficient  
    # -- SET PULSE PARAMETERS
    t0 = 0.0284                             # (ps) pulse duration
    P0 = np.abs(beta2)/t0/t0/gamma          # (W) pulse peak power
    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN 
    tMax = 1.                               # (ps) bound for time mesh 
    Nt = 1024                               # (-) number of sample points: t-axis
    zMax = 0.2                              # (m) upper limit for propagation routine
    Nz = 10000                              # (-) number of sample points: z-axis
    nSkip = 50                              # (-) number of z-steps to keep 

    # -- INITIALIZE COMPUTATIONAL DOMAIN
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)
    _z = np.linspace(0, zMax, Nz, endpoint=True)

    # -- DEFINE INTIAL REAL-VALUED FIELD
    A0 = np.sqrt(P0)/np.cosh(t/t0)

    # -- PROPAGATE FIELD
    z, Azt = SSFM_NSE_simple(_z, t, A0, beta2, gamma, nSkip)

    # -- COMPUTE FRACTIONAL ENERGY CHANGE BETWEEN INITIAL AND FINAL FIELD SLICE
    EIni = energy(t,Azt[0])
    EFin = energy(t,Azt[-1])
    print("# f_E =", np.abs(EIni-EFin)/EIni)

    # -- POSTPROCESS RESULTS
    figure_1a(z, t, Azt)

    ANSWER_a = "F_E should be 0, but there is an error for that"


def main_b():

    # -- SET FIBER PARAMETERS
    beta2 = -0.01276                        # (ps^2/m)
    gamma = 0.045                           # (1/W/m) nonlinear coefficient  
    # -- SET PULSE PARAMETERS
    t0 = 0.0284                             # (ps) pulse duration
    P0 = np.abs(beta2)/t0/t0/gamma          # (W) pulse peak power
    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN 
    tMax = 1.                               # (ps) bound for time mesh 
    Nt = 1024                               # (-) number of sample points: t-axis
    zMax = 0.5                              # (m) upper limit for propagation routine
    nSkip = 50                              # (-) number of z-steps to keep 

    # -- ANONYMOUS FUNCTION: EXACT SOLITON SOLUTION
    _AExact = lambda z,t: np.sqrt(P0)/np.cosh(t/t0)*np.exp(0.5j*gamma*P0*z)

    # -- ANONYMOUS FUNCTION: ROOT MEAN SQUARE ERROR
    _RMSError = lambda x,y: np.sqrt(np.sum(np.abs(x-y)**2)/x.size)

    # -- INITIALIZE COMPUTATIONAL DOMAIN
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)

    # -- SET INITIAL FIELD
    A0 = _AExact(0.,t)

    # -- RUN SIMULATION
    res = []
    for Nz in [2**n for n in range(7,14)]:
        print("# Nz = %d"%(Nz))
        _z = np.linspace(0, zMax, Nz, endpoint=True)
        z, Azt_m1 = SSFM_NSE_simple(_z, t, A0, beta2, gamma, nSkip)
        z, Azt_m2 = SSFM_NSE_symmetric(_z, t, A0, beta2, gamma, nSkip)
        AExact = _AExact(z[-1],t)

        # -- ACCUMULATE SIMULATION RESULTS
        res.append(( _z[1]-_z[0],              # z-stepsize
               _RMSError(AExact, Azt_m1[-1]),  # RMSE - simple splitting scheme
               _RMSError(AExact, Azt_m2[-1])   # RMSE - symmetric splitting scheme
              ))

    # -- POSTPROCESS RESULTS
    figure_1b(res)

    ANSWER_b = "The sym method this more accurate, they both get smaller error when dz decreaing"


def main_c():

    # -- SET FIBER PARAMETERS
    beta2 = -0.01276                        # (ps^2/m)
    gamma = 0.045                           # (1/W/m) nonlinear coefficient  
    # -- SET PULSE PARAMETERS
    t0 = 0.0284                             # (ps) pulse duration
    P0 = np.abs(beta2)/t0/t0/gamma          # (W) pulse peak power
    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN 
    tMax = 0.5                               # (ps) bound for time mesh 
    Nt = 1024*2                               # (-) number of sample points: t-axis
    zMax = 0.1                              # (m) upper limit for propagation routine
    Nz = 20000*4                             # (-) number of sample points: z-axis
    nSkip =100                              # (-) number of z-steps to keep 

    # -- INITIALIZE COMPUTATIONAL DOMAIN
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)
    _z = np.linspace(0, zMax, Nz, endpoint=True)

    # -- DEFINE INTIAL REAL-VALUED FIELD
    # TODO: MODIFY THE AMPLITUDE OF THE INITIAL CONDITION TO RESULT IN
    # A SOLITON OF GENERAL ORDER N (SEE LECTURE 06, SLIDE 24)
    N = 7
    A0 = np.sqrt(N**2*P0)/np.cosh(t/t0)

    # -- PROPAGATE FIELD
    z, Azt = SSFM_NSE_symmetric(_z, t, A0, beta2, gamma, nSkip)

    # -- COMPUTE FRACTIONAL ENERGY CHANGE BETWEEN INITIAL AND FINAL FIELD SLICE
    EIni = energy(t,Azt[0])
    EFin = energy(t,Azt[-1])
    print("# FRACTIONAL ENERGY CHANGE:", np.abs(EIni-EFin)/EIni)

    # -- POSTPROCESS RESULTS
    # -- CONSTRUCT FIGURE NAME (NOTE: MAKE SURE TO AMENDED THE METADATA VARIABLE __STUDENT_ID__)
    figName = "part1c_ID%s_Ns%g.png"%(__STUDENT_ID__, solitonOrder(t0, np.max(A0), beta2, gamma))
    # -- GENERATE OUTPUT FIGURE (NOTE: THIS WILLi BE SAVED IN YOUR CURRENT WORKING-DIRECTORY)
    figure_1c(
             z, t, Azt,             # simulation data (DO NOT CHANGE!)
             tLim=(-0.3,0.3),         # TODO: adjust limits of t-axis (left subfigure) to improve figure!
             wLim=(-6000.,6000),    # TODO: adjust limits of omega-axis (right subfigure) to improve figure!
             oName=figName          # name of the generated figure (DO NOT CHANGE!)
             )


if __name__ == "__main__":
    #main_a()
    main_b()
    #main_c()

# EOF: main_ex04_part1_ID.py
