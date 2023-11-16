
import numpy as np
import matplotlib.pyplot as plt
#import ipywidgets as widgets
from IPython.display import display




def split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz):
    dx = x[1] - x[0]
    k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    
    A1 = A1_init.copy()
    A2 = A2_init.copy()
    A3 = A3_init.copy()
    
    A1_results = [A1]
    A2_results = [A2]
    A3_results = [A3]

    for i in range(len(z) - 1):
        
        # Linear step (Fourier domain)


        #print(A1 - A2)
        

        
        # Nonlinear step (z-domain update)   
        A1_n = A1 + -1j * gamma1 * A3 * np.conj(A2)  * dz
        A2_n = A2 + -1j * gamma2 * A3 * np.conj(A1)  * dz
        a3 = 1j * A3 * k_hat
        A3_n = A3 + (-1j * gamma3 * A1 * A2 - a3 ) * dz
        #print("a3 = ",a3)

        A1 = A1_n
        A2 = A2_n
        A3 = A3_n


        #add to list
        
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

D = 2
D1, D2, D3 = 1/k1/D, 1/k2/D, 1/k3/D
gamma = 10
gamma1, gamma2, gamma3 = 10,10,10


x = np.linspace(-20 , 20 , 500)
z = np.linspace(0, 2000, 5000)
dz = z[1] - z[0]

# Initial conditions
theta2 = 2.9
E1 = 0.6 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1
x0 = 8
A1_init = E1*np.exp(-1*x**2/a1**2)
A2_init = E2*np.exp(-1*(x - x0)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8
A3_init = np.exp(-x**2)*0

k_hat = -1*k1*k2*theta2*theta2/2.0/k3
theta_cr = np.power(4*gamma2*gamma3*k3/k1/k2,1/4)*np.square(E1)
theta_min = 2 * x0 / k1 / a1**2
print("k_hat is",k_hat)
print("theta_cr is",theta_cr)
print("theta_min is",theta_min)


def power():
    # Apply split-step Fourier method
    A1_results, A2_results, A3_results = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
    # Calculate absolute values of the fields and transpose them
    A1_abs = np.square(np.abs(A1_results).T) 
    A2_abs = np.square(np.abs(A2_results).T)
    A3_abs = np.square(np.abs(A3_results).T)
    p1 = np.sum(A1_abs,axis=0)
    p2 = np.sum(A2_abs,axis=0)
    p3 = np.sum(A3_abs,axis=0)
    print(p2[0])
    p20 = p2[0]
    p2 = p2/p20
    p3 = p3/p20
    #plt.plot(z,p3)
    plt.plot(z,p2)
    plt.ylim(0, 2)
    plt.show()

#power()

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
    plt.title('Absolute value of A1 field')
    plt.show()

#plot_A1()

def plot_A2():
    # Apply split-step Fourier method
    _, A2_results, _ = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
    # Calculate absolute values of A2 and transpose it
    A2_abs = np.abs(A2_results).T
    print(A2_abs.shape)
    # Plot the absolute value of A2 as a function of x and z
    plt.imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='jet')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.colorbar(label='|A2|')
    plt.title('Absolute value of A2 field')
    plt.show()
    
#plot_A2()

def plot_diff():

    A1_results, A2_results, A3_results = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
    # Calculate absolute values of the fields and transpose them
    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T

    A2_abs =  A1_abs - A2_abs
    # Plot the absolute value of A2 as a function of x and z
    plt.imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='jet')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.colorbar(label='|A2|')
    plt.title('Absolute value of A2 field')
    plt.show()

#plot_diff()

def plot_A2_gamma2():
    # Define gamma2 values to plot
    gamma2_list = np.logspace(-2,2,20)

    # Set up subplot grid
    n_plots = len(gamma2_list)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    # Loop over gamma2 values and plot each one
    for i, gamma2 in enumerate(gamma2_list):
        # Apply split-step Fourier method
        _, A2_results, _ = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)

        # Calculate absolute values of A2 and transpose it
        A2_abs = np.abs(A2_results).T

        # Determine subplot index
        row_idx = i // n_cols
        col_idx = i % n_cols

        # Plot the absolute value of A2 as a function of x and z
        axs[row_idx, col_idx].imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='jet')
        axs[row_idx, col_idx].set_xlabel('z')
        axs[row_idx, col_idx].set_ylabel('x')
        axs[row_idx, col_idx].set_title(f'gamma2 = {gamma2:.0e}')
        axs[row_idx, col_idx].set_aspect('equal')
        axs[row_idx, col_idx].set_xlim([z.min(), z.max()])
        axs[row_idx, col_idx].set_ylim([x.min(), x.max()])

    # Add colorbar
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(axs[0, 0].images[0], cax=cbar_ax, label='|A2|')

    # Add overall title
    fig.suptitle('Absolute value of A2 field for different gamma2 values')
    plt.show()



#plot_A2_gamma2()


def plot_A2_theta2():
    # Define gamma2 values to plot
    theta2_list = np.linspace(1,10,6)

    # Set up subplot grid
    n_plots = len(theta2_list)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5, 5))

    # Loop over gamma2 values and plot each one
    for i, theta2 in enumerate(theta2_list):
        # Apply split-step Fourier method
        theta2 = theta2
        A2_init = E2*np.exp(-1*(x - x0)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8
        _, A2_results, _ = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)

        # Calculate absolute values of A2 and transpose it
        A2_abs = np.abs(A2_results).T

        # Determine subplot index
        row_idx = i // n_cols
        col_idx = i % n_cols

        # Plot the absolute value of A2 as a function of x and z
        axs[row_idx, col_idx].imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='jet')
        axs[row_idx, col_idx].set_xlabel('z')
        axs[row_idx, col_idx].set_ylabel('x')
        axs[row_idx, col_idx].set_title(f'theta2 = {theta2:.0e}')
        axs[row_idx, col_idx].set_aspect('equal')
        axs[row_idx, col_idx].set_xlim([z.min(), z.max()])
        axs[row_idx, col_idx].set_ylim([x.min(), x.max()])

    # Add colorbar
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(axs[0, 0].images[0], cax=cbar_ax, label='|A2|')

    # Add overall title
    fig.suptitle('Absolute value of A2 field for different gamma2 values')
    plt.show()



#plot_A2_theta2()

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
    plt.title('Absolute values of A1 (Blue), A2 (Red), and A3 (Green) fields ')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.5, label='|A1|'),
                       Patch(facecolor='red', alpha=0.5, label='|A2|'),
                       Patch(facecolor='green', alpha=0.5, label='|A3|')]
    plt.legend(handles=legend_elements)
    
    plt.show()

plot_all()

def save_all():
    # Apply split-step Fourier method
    A1_results, A2_results, A3_results = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
    # Calculate absolute values of the fields and transpose them
    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T
    
    np.savetxt("A1.txt",A1_abs)
    np.savetxt("A2.txt",A2_abs)
    np.savetxt("A3.txt",A3_abs)
#save_all()

def reload():

    A1_abs = np.loadtxt("A1.txt")
    A2_abs = np.loadtxt("A2.txt")
    A3_abs = np.loadtxt("A3.txt")
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

#reload()