
import numpy as np
import matplotlib.pyplot as plt


def split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz):
    dx = x[1] - x[0]
    k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    
    A1 = A1_init.copy()
    A2 = A2_init.copy()
    A3 = A3_init.copy()
    
    A1_results = [A1]
    A2_results = [A2]
    A3_results = [A3]

    for _ in range(len(z) - 1):
        # Linear step (Fourier domain)
        A1_k = np.fft.fft(A1)
        A2_k = np.fft.fft(A2)
        A3_k = np.fft.fft(A3)

        A1_k *= np.exp(1j * D1 * k_x**2 * dz)
        A2_k *= np.exp(1j * D2 * k_x**2 * dz)
        A3_k *= np.exp(1j * D3 * k_x**2 * dz)

        A1 = np.fft.ifft(A1_k)
        A2 = np.fft.ifft(A2_k)
        A3 = np.fft.ifft(A3_k)

        # Nonlinear step (z-domain update)
        A1 += -1j * gamma1 * A3 * np.conj(A2) * np.exp(1j * k_hat * dz) * dz
        A2 += -1j * gamma2 * A3 * np.conj(A1) * np.exp(1j * k_hat * dz) * dz
        A3 += -1j * gamma3 * A1 * A2 * np.exp(-1j * k_hat * dz) * dz
        
        A1_results.append(A1.copy())
        A2_results.append(A2.copy())
        A3_results.append(A3.copy())

    return np.array(A1_results), np.array(A2_results), np.array(A3_results)

# Example parameters
k1, k2, k3 = 1,2,3
k_parameter = 100 #100
k1 = k1*k_parameter
k1 = k2*k_parameter
k1 = k3*k_parameter

D1, D2, D3 = 1/k1/2, 1/k2/2, 1/k3/2
gamma1, gamma2, gamma3 = 10, 10, 10


x = np.linspace(-10, 10, 512)
z = np.linspace(0, 2, 100)
dz = z[1] - z[0]

# Initial conditions
theta2 = 0.5
E1 = 0.4 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1
A1_init = E1*np.exp(-x**2/a1**2)
A2_init = E2*np.exp(-(x - 8)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8
A3_init = np.exp(-x**2)*0

k_hat = -1*k1*k2*theta2*theta2/2.0/k3

def plot_A1():
    # Apply split-step Fourier method
    A1_results, _, _ = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
    # Calculate absolute values of A1 and transpose it
    A1_abs = np.abs(A1_results).T
    
    # Plot the absolute value of A1 as a function of x and z
    plt.imshow(A1_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='jet')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.colorbar(label='|A1|')
    plt.title('Absolute value of A1 field (rotated 90°)')
    plt.show()

#plot_A1()

def plot_A2():
    # Apply split-step Fourier method
    _, A2_results, _ = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
    # Calculate absolute values of A2 and transpose it
    A2_abs = np.abs(A2_results).T
    
    # Plot the absolute value of A2 as a function of x and z
    plt.imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='jet')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.colorbar(label='|A2|')
    plt.title('Absolute value of A2 field (rotated 90°)')
    plt.show()
#plot_A2()

def plot_all():
    # Apply split-step Fourier method
    A1_results, A2_results, A3_results = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
    # Calculate absolute values of the fields and transpose them
    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T
    
    # Plot the absolute value of A1, A2, and A3 as a function of x and z
    plt.imshow(A1_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Blues', alpha=1)
    plt.imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Reds', alpha=0.5)
    plt.imshow(A3_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Greens', alpha=0.5)
    
    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('Absolute values of A1 (Blue), A2 (Red), and A3 (Green) fields (rotated 90°)')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.5, label='|A1|'),
                       Patch(facecolor='red', alpha=0.5, label='|A2|'),
                       Patch(facecolor='green', alpha=0.5, label='|A3|')]
    plt.legend(handles=legend_elements)
    
    plt.show()

plot_all()

