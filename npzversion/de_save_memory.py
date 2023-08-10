
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

directory = 'C:\\Users\\DevlandZ\\Desktop\\Study\\master thesis\\split_step_fourier_method\\data'
os.chdir(directory)

def ssfm_once(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, k_x, dz):
   
    A1 = A1_init.copy()
    A2 = A2_init.copy()
    A3 = A3_init.copy()

    # Nonlinear step (z-domain update)

    A1_n = A1 + -1j * gamma1 * A3 * np.conj(A2)  * dz
    A2_n = A2 + -1j * gamma2 * A3 * np.conj(A1)  * dz
    a3 = 1j * A3 * k_hat
    A3_n = A3 + (-1j * gamma3 * A1 * A2 + a3 ) * dz
    #print("a3 = ",a3)

    A1 = A1_n
    A2 = A2_n
    A3 = A3_n
    
    # Linear step (Fourier domain)
    FT = np.fft.ifft
    IFT = np.fft.fft
    A1 =  IFT(np.exp(1j  * D1 * k_x**2 * dz*0.5)*FT(A1))
    A2 =  IFT(np.exp(1j  * D2 * k_x**2 * dz*0.5)*FT(A2))
    A3 =  IFT(np.exp(1j  * D3 * k_x**2 * dz*0.5)*FT(A3))



    return np.array(A1), np.array(A2), np.array(A3)

#  parameters
k1, k2, k3 = 1,2,3
k_parameter = 1  #100
k1 = k1*k_parameter
k1 = k2*k_parameter
k1 = k3*k_parameter

D = 10
D1, D2, D3 = 1/k1/D, 1/k2/D, 1/k3/D
gamma = 10
gamma1, gamma2, gamma3 = 10,10,10


x = np.linspace(-10, 10, 1000)
z = np.linspace( 0,50, 50000)
dz = z[1] - z[0]
dx = x[1] - x[0]
k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)


# Initial conditions
theta2 = 2
E1 = 0.4 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1

z_size = 5000
k_hat = -1*k1*k2*theta2*theta2/2.0/k3
print(k_hat)




def save_all():
    theta2 = 2
    E1 = 0.4 # 0.2 0.6
    E2 = 1e-2 #1e-2
    a1 = 3 #3
    a2 = 1 #1



    x0 = 8
    A1_init = E1*np.exp(-x**2/a1**2)
    A2_init = E2*np.exp(-(x - x0)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8
    A3_init = np.exp(-x**2)*0



    # Apply split-step Fourier method
    A1_results = [A1_init]
    A2_results = [A2_init]
    A3_results = [A3_init]
    index = -1

     
    
    for i in range(1,len(z)):
        A1, A2, A3 = ssfm_once(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, k_x, dz)
        
        if i % z_size == 1:
            index += 1
            results = {
                "z"  : z,
                "A1" : np.array(A1_results),
                "A2" : np.array(A2_results),
                "A3" : np.array(A3_results)
            }
            f_name = f'theta2={theta2}_E1={E1}_{index}'
            np.savez_compressed(f_name, **results)
            
            A1_results = [A1_init]
            A2_results = [A2_init]
            A3_results = [A3_init]
        A1_results.append(A1.copy())
        A2_results.append(A2.copy())
        A3_results.append(A3.copy())
        A1_init, A2_init, A3_init = A1, A2, A3
    print(index)


   
#save_all()

def load(theta2 = 2,E1 = 0.4):
    def fetch_data(f_name):
        dat = np.load(f_name)
        return dat['z'], dat['A1'],dat['A2'],dat['A3']
    

    index = 9


    f_name = f'theta2={theta2}_E1={E1}_{0}.npz'
    z,A1_results,A2_results,A3_results   = fetch_data(f_name)
    for i in range(1, index):
        f_name = f'theta2={theta2}_E1={E1}_{i}.npz'
        z,A1_array,A2_array,A3_array   = fetch_data(f_name)
        A1_results = np.concatenate((A1_results, A1_array), axis=0)
        A2_results = np.concatenate((A2_results, A2_array), axis=0)
        A3_results = np.concatenate((A3_results, A3_array), axis=0)




    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T
    
    # Plot the absolute value of A1, A2, and A3 as a function of x and z
    plt.imshow(A1_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Blues', alpha=1)
    plt.imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Reds', alpha=0.5)
    plt.imshow(A3_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Greens', alpha=0.5)
    
    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('Absolute values of A1 (Blue), A2 (Red), and A3 (Green) fields ')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.5, label='|A1|'),
                       Patch(facecolor='red', alpha=0.5, label='|A2|'),
                       Patch(facecolor='green', alpha=0.5, label='|A3|')]
    plt.legend(handles=legend_elements)
    
    plt.show()

    return A1_abs,A2_abs,A3_abs

load()

def reflection():
    # Apply split-step Fourier method
    _, A2_results, _ = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)

    # Calculate absolute values of A2 and transpose it
    A2_abs = np.abs(A2_results).T
    print(A2_abs.shape)
    # for array shape of (1000,50000)
    total_power = np.sum(np.square(A2_abs[649:1000, 4999:10000]))
    reflect_power = np.sum(np.square(A2_abs[649:1000, 31999:37000]))
    R = reflect_power/total_power
    print("reflection is ",R)

reflection()


