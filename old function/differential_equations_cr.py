
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


x = np.linspace(-20, 20, 1000)
z = np.linspace( 0,100, 30000)
dz = z[1] - z[0]

# Initial conditions
theta2 = 2
E1 = 0.1 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1
x0 = 8
A1_init = E1*np.exp(-x**2/a1**2)
A2_init = E2*np.exp(-(x - x0)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8
A3_init = np.exp(-x**2)*0

k_hat = -1*k1*k2*theta2*theta2/2.0/k3
print(k_hat)


def impulse():
    A1_results, A2_results, A3_results = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    diff = np.diff(A1_results, axis=1)
    product = np.imag(np.conjugate(A1_results[:,:-1])* diff)
    I1 = product.sum(axis=1)

    diff = np.diff(A2_results, axis=1)
    product = np.imag(np.conjugate(A2_results[:,:-1])* diff)
    I2 = product.sum(axis=1)

    diff = np.diff(A3_results, axis=1)
    product = np.imag(np.conjugate(A3_results[:,:-1])* diff)
    I3 = product.sum(axis=1)

    I_sum = I1 + I2 + I3

    I1 = I1 / I_sum[0]
    I2 = I2 / I_sum[0]
    I3 = I3 / I_sum[0]
    I_sum = I_sum  / I_sum[0]



    plt.plot(z,I1,label='I1')
    plt.plot(z,I2,label='I2')
    plt.plot(z,I3,label='I3')
    plt.plot(z,I_sum,label='sum')
    plt.legend()
    plt.show()

#impulse()


def focused():
    '''
    l_dif = k1 * a1**2 *0.5
    m = theta2 * l_dif / (2 * x0)
    R1 = l_dif * (m + np.sqrt(m**2 - 1))
    dz1 = 2 * a2 / theta2
    N = 2
    dz2 = 2 * l_dif * np.sqrt(np.power(N,-4) - 1)/ (1 + (l_dif / R1)**2)
    '''
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


    x = np.linspace(-20, 20, 500)
    z = np.linspace( 0,60, 10000)
    dz = z[1] - z[0]

    # Initial conditions
    theta2 = 2
    E1 = 0.4 # 0.2 0.6
    E2 = 1e-2 #1e-2
    a1 = 3 #3
    a2 = 1 #1
    x0 = 8
    A1_init = E1*np.exp(-x**2/a1**2)
    A2_init = E2*np.exp(-(x - x0)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8
    A3_init = np.exp(-x**2)*0

    k_hat = -1*k1*k2*theta2*theta2/2.0/k3
    print(k_hat)
    dx = x[1] - x[0]
    k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    
    
    A1 = A1_init.copy()
    A2 = A2_init.copy()
    A3 = A3_init.copy()
    
    A1_results = [A1]
    A2_results = [A2]
    A3_results = [A3]

    for i in range(len(z) - 1):

        # Nonlinear step (z-domain update)
        #A1_n = A1 + -1j * gamma1 * A3 * np.conj(A2)  * dz
        A2_n = A2 + -1j * gamma2 * A3 * np.conj(A1)  * dz
        a3 = 1j * A3 * k_hat
        A3_n = A3 + (-1j * gamma3 * A1 * A2 + a3 ) * dz
        #print("a3 = ",a3)

        #A1 = A1_n
        A2 = A2_n
        A3 = A3_n
        
        # Linear step (Fourier domain)
        FT = np.fft.ifft
        IFT = np.fft.fft
        #A1 =  IFT(np.exp(1j  * D1 * k_x**2 * dz*0.5)*FT(A1))
        A2 =  IFT(np.exp(1j  * D2 * k_x**2 * dz*0.5)*FT(A2))
        A3 =  IFT(np.exp(1j  * D3 * k_x**2 * dz*0.5)*FT(A3))


        power = 0.4**2 * 3
        E1 = 2/(np.abs(z[i] - 27)/70+0.5)*0.4*1.0
        a1 = power / E1**2 *0.1
        A1 = E1*np.exp(-x**2/a1**2)
        A1_results.append(A1.copy())
        A2_results.append(A2.copy())
        A3_results.append(A3.copy())



    
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




def power():
    # Apply split-step Fourier method
    A1_results, A2_results, A3_results = split_step_fourier_method_1D(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, x, z, dz)
    
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
    #plt.plot(z,p3)
    plt.plot(z,p2)
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

#plot_all()



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

#reflection()

def theta_cr():
    E1_list = np.linspace(0.2, 0.6, 20)
    similar = []
    for E1 in E1_list:

        
        theta_cr = np.power(4 * gamma2 * gamma3 * k3 / k1 / (k2**2), 1/4) * E1**0.5
        theta_similar = theta_cr / theta2
        similar.append(theta_similar)
    
    print(theta2,2*x0/(k1 * a1**2))
    R1 =  [0.18277668674356337, 0.19174758380417942, 0.20671423158032723, 0.2523866667531735, 0.3691838035643133, 0.5517523955818884, 0.7285780403538605, 0.8463380148268628, 0.9083859755010932, 0.9379067029114851, 0.9522075993899696, 0.9598560837713306, 0.9645478064516619, 0.9678679949487943, 0.9705019858137003, 0.9727415515946822, 0.9747116170054366, 0.9764805663416117, 0.9780852433579119, 0.9795507183193806]
    # Plotting
    plt.plot(E1_list, similar, label=r'$\theta_{cr}/\theta_{2}$')  # Using LaTeX for Theta_cr
    plt.plot(E1_list, R1, label='Reflection')

    # Adding a horizontal line at y = 1 and a vertical line at the corresponding E1 value
    plt.axhline(y=1, color='r', linestyle='--')
    plt.axvline(x=0.40, color='r', linestyle='--')

    # Adding legends
    plt.legend()

    # Labeling the horizontal line with LaTeX and positioning it closer to the line
    plt.text(0.5 * max(E1_list), 1.02, r'$\theta_{2}$', verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=10)

    # Show the plot
    plt.show()
theta_cr()
