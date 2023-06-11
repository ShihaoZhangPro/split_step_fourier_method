
import numpy as np
import matplotlib.pyplot as plt
#import ipywidgets as widgets
from IPython.display import display
import h5py


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
    A1_k = np.fft.fft(A1)
    A2_k = np.fft.fft(A2)
    A3_k = np.fft.fft(A3)

    A1_k *= np.exp(1j  * D1 * k_x**2 * dz)
    A2_k *= np.exp(1j  * D2 * k_x**2 * dz)
    A3_k *= np.exp(1j  * D3 * k_x**2 * dz)

    A1 = np.fft.ifft(A1_k)
    A2 = np.fft.ifft(A2_k)
    A3 = np.fft.ifft(A3_k)



    return np.array(A1), np.array(A2), np.array(A3)

# Example parameters
k1, k2, k3 = 1,2,3
k_parameter = 1 #100
k1 = k1*k_parameter
k1 = k2*k_parameter
k1 = k3*k_parameter

D = 10
D1, D2, D3 = 1/k1/D, 1/k2/D, 1/k3/D
gamma = 10
gamma1, gamma2, gamma3 = 1e1,1e02,1e1

x = np.linspace(-20, 20, 5000)
z = np.linspace(0, 30, 70000)
dz = z[1] - z[0]
dx = x[1] - x[0]
k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
z_size = 10000 


theta2 = 1.9
E1 = 0.4 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1

k_hat = -1*k1*k2*theta2*theta2/2.0/k3
print(k_hat)




def save_all():
    
    A1_init = E1*np.exp(-x**2/a1**2)
    A2_init = E2*np.exp(-(x - 8)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8
    A3_init = np.exp(-x**2)*0



    # Apply split-step Fourier method
    A1_results = [A1_init]
    A2_results = [A2_init]
    A3_results = [A3_init]
    index = -1
    with h5py.File('A1.h5', 'w') as hf1,h5py.File('A2.h5', 'w') as hf2,h5py.File('A3.h5', 'w') as hf3:
        for i in range(1,len(z)):
            A1, A2, A3 = ssfm_once(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, k_x, dz)
            #print("this is the shape and length of A1",A1.shape,A1.size)
            if i % z_size == 1:
                index += 1
                print("this is the length of A1",len(A1_results))
                hf1.create_dataset(f'A1_0.6_{index}', data=A1_results)
                hf2.create_dataset(f'A2_0.6_{index}', data=A2_results)
                hf3.create_dataset(f'A3_0.6_{index}', data=A3_results)
                A1_results = [A1_init]
                A2_results = [A2_init]
                A3_results = [A3_init]
            A1_results.append(A1.copy())
            A2_results.append(A2.copy())
            A3_results.append(A3.copy())
            A1_init, A2_init, A3_init = A1, A2, A3
    print(index) 

   
#save_all()

def load():

    index = 6



    with h5py.File('A1.h5', 'r') as hf1:
        A1_results = hf1['A1_0.6_0'][:] 
        for i in range(1,index): 
            array = hf1[f'A1_0.6_{i}'][:] 
            A1_results = np.concatenate((A1_results, array), axis=0)
    with h5py.File('A2.h5', 'r') as hf2:
        A2_results = hf2['A2_0.6_0'][:] 
        for i in range(1,index): 
            array = hf2[f'A2_0.6_{i}'][:] 
            A2_results = np.concatenate((A2_results, array), axis=0)
    with h5py.File('A3.h5', 'r') as hf3:
        A3_results = hf3['A3_0.6_0'][:] 
        for i in range(1,index): 
            array = hf3[f'A3_0.6_{i}'][:] 
            A3_results = np.concatenate((A3_results, array), axis=0)


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

load()


def power():
    index = 6
    with h5py.File('A1.h5', 'r') as hf1:
        A1_results = hf1['A1_0.6_0'][:] 
        for i in range(1,index): 
            array = hf1[f'A1_0.6_{i}'][:] 
            A1_results = np.concatenate((A1_results, array), axis=0)
    with h5py.File('A2.h5', 'r') as hf2:
        A2_results = hf2['A2_0.6_0'][:] 
        for i in range(1,index): 
            array = hf2[f'A2_0.6_{i}'][:] 
            A2_results = np.concatenate((A2_results, array), axis=0)
    with h5py.File('A3.h5', 'r') as hf3:
        A3_results = hf3['A3_0.6_0'][:] 
        for i in range(1,index): 
            array = hf3[f'A3_0.6_{i}'][:] 
            A3_results = np.concatenate((A3_results, array), axis=0)
    
    # Calculate absolute values of the fields and transpose them
    A1_abs = np.square(np.abs(A1_results).T)
    A2_abs = np.square(np.abs(A2_results).T)
    A3_abs = np.square(np.abs(A3_results).T)
    p1 = sum(A1_abs)
    p2 = sum(A2_abs)
    p3 = sum(A3_abs)
    p20 = p2[0]
    p2 = p2/p20
    p1 = p1/p20
    p1 = p3/p20
    plt.plot(z,p3)
    plt.show()

#power()