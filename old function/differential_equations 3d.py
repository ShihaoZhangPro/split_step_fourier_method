
import numpy as np
import matplotlib.pyplot as plt
#import ipywidgets as widgets
from IPython.display import display


def split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz):
    dx = x[1] - x[0]
    dy = dx
    k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    k_y = 2 * np.pi * np.fft.fftfreq(len(x), d=dy)  # Assuming square grid with same number of points in x and y

    k_x, k_y = np.meshgrid(k_x, k_y)

    
    A1 = A1_init.copy()
    A2 = A2_init.copy()
    A3 = A3_init.copy()
    
    A1_results = [A1]
    A2_results = [A2]
    A3_results = [A3]

    for i in range(len(z) - 1):

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
        FT2 = np.fft.fft2
        IFT2 = np.fft.ifft2

        A1 = IFT2(np.exp(1j * D1 * (k_x**2 + k_y**2) * dz * 0.5) * FT2(A1))
        A2 = IFT2(np.exp(1j * D2 * (k_x**2 + k_y**2) * dz * 0.5) * FT2(A2))
        A3 = IFT2(np.exp(1j * D3 * (k_x**2 + k_y**2) * dz * 0.5) * FT2(A3))
        # Nonlinear step (z-domain update)
        
        A1_results.append(A1.copy())
        A2_results.append(A2.copy())
        A3_results.append(A3.copy())

    return np.array(A1_results), np.array(A2_results), np.array(A3_results)


# Example parameters
k1, k2, k3 = 1,2,3
k_parameter = 1  #100
k1 = k1*k_parameter
k1 = k2*k_parameter
k1 = k3*k_parameter

D = 10
D1, D2, D3 = 1/k1/D, 1/k2/D, 1/k3/D
gamma = 10
gamma1, gamma2, gamma3 = 10,10,10


x = np.linspace(-20, 20, 64)
y = x
z = np.linspace( 0,100, 50000)
dz = z[1] - z[0]

# Initial conditions
theta2 = 2
E1 = 0.4 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1
x0 = 8
X, Y = np.meshgrid(x, y)  # create a 2D grid

# A1
A1_init = E1 * np.exp(-(X**2 + Y**2) / a1**2)

# A2
A2_init = E2 * np.exp(-((X - x0)**2 + Y**2) / a2**2 + 1j * k2 * theta2 * X)

# A3
A3_init = np.exp(-(X**2 + Y**2)) * 0

k_hat = -1*k1*k2*theta2*theta2/2.0/k3
print(k_hat)







def plot_all():
    # Apply split-step Fourier method
    A1_results, A2_results, A3_results = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)




    # Calculate absolute values of the fields and transpose them
    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T


    data = A2_abs
    slice_y = data[:, 50, :]
    plt.imshow(slice_y, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Blues', alpha=1)
    plt.title('XZ slice at y=50')
    plt.colorbar()
    plt.show()
    '''
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
    '''
plot_all()

