
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import matplotlib.gridspec as gridspec

# save the data to the data file
current_directory = os.path.dirname(os.path.abspath(__file__))
main_directory = os.path.join(current_directory, '..\..')
data_directory = os.path.join(main_directory, 'data')
os.chdir(data_directory)
print(data_directory)

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


x = np.linspace(-20, 20, 1024)
z = np.linspace( 0,70, 50000)
dz = z[1] - z[0]
dx = x[1] - x[0]
k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)


# Initial conditions
theta2 = 2
E1 = 0.4 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1

z_size = 10000
k_hat = -1*k1*k2*theta2*theta2/2.0/k3
print(k_hat)




def save_data(theta2 = 2,E1 = 0.2):
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

     
    
    for i in range(0,len(z)+1):
        
        
        if i % z_size == 0:
            print(len(A1_results))
            #print(i)
            index += 1
            results = {
                "z"  : z,
                "A1" : np.array(A1_results),
                "A2" : np.array(A2_results),
                "A3" : np.array(A3_results)
            }
            f_name = f'theta2={theta2}_E1={E1}_{index}.npz'
            np.savez_compressed(f_name, **results)
            
            A1_results = []
            A2_results = []
            A3_results = []
        A1, A2, A3 = ssfm_once(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, k_x, dz)
        A1_results.append(A1.copy())
        A2_results.append(A2.copy())
        A3_results.append(A3.copy())
        A1_init, A2_init, A3_init = A1, A2, A3
    print(index)


   
#save_data(theta2 = 2,E1 = 0.1)

def load(theta2 = 2,E1 = 0.4,index = 5):
    def fetch_data(f_name):
        dat = np.load(f_name)
        return dat['z'], dat['A1'],dat['A2'],dat['A3']
    



    f_name = f'theta2={theta2}_E1={E1}_{1}.npz'
    z,A1_results,A2_results,A3_results   = fetch_data(f_name)
    for i in range(2, index+1):
        f_name = f'theta2={theta2}_E1={E1}_{i}.npz'
        z,A1_array,A2_array,A3_array   = fetch_data(f_name)
        #print(A1_results.shape)
        A1_results = np.concatenate((A1_results, A1_array), axis=0)
        A2_results = np.concatenate((A2_results, A2_array), axis=0)
        A3_results = np.concatenate((A3_results, A3_array), axis=0)




    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T

    print(A1_abs.shape)
    


    return A1_abs,A2_abs,A3_abs

#load()

def plot_all():
    
    A1_abs,A2_abs,A3_abs = load(theta2 = 2,E1 = 0.2)



    # Display A1
    fig1, ax1 = plt.subplots()
    cax1 = ax1.imshow(A1_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Blues', alpha=1)
    ax1.set_ylabel("Y-Axis Label")
    ax1.set_xticks([])  # Hide x-axis labels and ticks for A1
    plt.colorbar(cax1)


    # Display A2
    fig2, ax2 = plt.subplots()
    cax2 = ax2.imshow(A2_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Reds', alpha=0.5)
    ax2.set_ylabel("Y-Axis Label")
    ax2.set_xticks([])  # Hide x-axis labels and ticks for A2
    plt.colorbar(cax2)


    # Display A3
    fig3, ax3 = plt.subplots()
    cax3 = ax3.imshow(A3_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Greens', alpha=0.5)
    ax3.set_ylabel("Y-Axis Label")
    ax3.set_xlabel("X-Axis Label")  # Show x-axis labels and ticks only for A3
    plt.colorbar(cax3)

    plt.show()


#plot_all()

def get_center(A):


    squared_array = A ** 2

    # Compute the weighted version of the squared array for columns
    weighted_array = np.multiply(squared_array, x.reshape(-1,1))

    # Compute the total weight for each column
    total_weight_per_column = np.sum(weighted_array, axis=0)

    # Compute the total intensity for each column
    total_intensity_per_column = np.sum(squared_array, axis=0)

    # Compute the weighted center position for each column
    center_positions_columns = total_weight_per_column / total_intensity_per_column

    return(center_positions_columns)

#get_center()

def plot_center():
    
    A1_abs,A2_abs,A3_abs = load(theta2 = 2,E1 = 0.1)

    A1_center =get_center(A1_abs) 
    A2_center =get_center(A2_abs)
    A3_center =get_center(A3_abs)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 9))  # 3 rows, 1 column
    # Display A1
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    cax1 = ax1.imshow(A1_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Blues', alpha=1)
    ax1.axhline(y=0, color='black', linewidth=2,linestyle='--')
    ax1.plot(z,A1_center, color='red', linewidth=2,label='center position')
    ax1.legend(prop={'size': 16},loc='lower right')
    ax1.set_ylabel("x",fontsize=20)
    ax1.set_xticklabels([])  # Hide x-axis labels and ticks for A1
    plt.colorbar(cax1)


    # Display A2
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    cax2 = ax2.imshow(A2_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Oranges', alpha=0.5)
    ax2.axhline(y=0, color='black', linewidth=2,linestyle='--')
    ax2.plot(z,A2_center, color='red', linewidth=2,label='center position')
    ax2.legend(prop={'size': 16},loc='lower right')
    ax2.set_ylabel("x",fontsize=20)
    ax2.set_xticklabels([])  # Hide x-axis labels and ticks for A2
    plt.colorbar(cax2)


    # Display A3
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    cax3 = ax3.imshow(A3_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Greens', alpha=0.5)
    ax3.axhline(y=0, color='black', linewidth=2,linestyle='--')

    # Create the mask
    mask = A3_center < 3

    # Apply the mask
    z_masked = np.array(z)[mask]
    A3_center_masked = A3_center[mask]

    # Plot the masked values
    ax3.plot(z_masked, A3_center_masked, color='red', linewidth=2, label='center position')


    ax3.legend(prop={'size': 16},loc='lower right')
    ax3.set_ylabel("x",fontsize=20)
    ax3.set_xlabel("z",fontsize=20)  # Show x-axis labels and ticks only for A3
    plt.colorbar(cax3)


    plt.tight_layout()
    plt.show()


#plot_center()

def plot_center_inone(theta2 = 2,E1 = 0.4):

    A1_abs,A2_abs,A3_abs = load(theta2 = theta2,E1 = E1)

    A1_center =get_center(A1_abs) 
    A2_center =get_center(A2_abs)
    A3_center =get_center(A3_abs)

    fig = plt.figure(figsize=(15, 9))

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.05)

    def add_colorbar(ax, cax_data):
        # Define custom position and size for colorbar
        left, bottom, width, height = ax.get_position().bounds
        cax = fig.add_axes([left + width + 0.003, bottom, 0.008, height / 2])
        fig.colorbar(cax_data, cax=cax)
        cbar = fig.colorbar(cax_data, cax=cax)
        cbar.ax.tick_params(labelsize=15)

    def add_title(ax, title_text):
        # Get the axis position and dimensions
        left, bottom, width, height = ax.get_position().bounds
        # Place title text at the right top corner of the axis
        ax.text(left + width + 0.04, bottom + height, title_text, 
                ha='right', va='top', transform=fig.transFigure, fontsize=20)
    
    def hide_extreme_ytick_labels(ax):
        yticks = ax.get_yticklabels()
        if yticks:
            yticks[0].set_visible(False)  # Hide smallest label
            yticks[-1].set_visible(False)  # Hide largest label

    def adjust_axis_properties(ax):
        # Adjust spines
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        
        # Adjust tick widths
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)


    def add_index(ax, text="(a)"):
        # Get the position of the axis
        left, bottom, width, height = ax.get_position().bounds
        # Add text to the top-left corner of the ax, adjusting position as needed
        ax.text(left, bottom + height, text, 
                color='white', 
                backgroundcolor='black', 
                verticalalignment='top',
                horizontalalignment='left', 
                transform=ax.figure.transFigure,
                fontsize=20)

    # Data1
    ax1 = plt.subplot(gs[0])
    
    cax1 = ax1.imshow(A1_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Blues', alpha=1)
    ax1.axhline(y=0, color='black', linewidth=2,linestyle='--')
    ax1.plot(z,A1_center, color='red', linewidth=2,label='center position')
    ax1.legend(prop={'size': 16},loc='lower right')
    ax1.set_ylabel("x",fontsize=20)
    ax1.set_ylim(-10, 10)
    ax1.set_xticklabels([])  
    hide_extreme_ytick_labels(ax1)
    add_title(ax1, title_text = r"$|A1|$")
    add_colorbar(ax1, cax1)
    adjust_axis_properties(ax1)
    add_index(ax1, text="(a)")
    ax1.tick_params(axis='both', labelsize=20)


    # Data2
    ax2 = plt.subplot(gs[1])

    cax2 = ax2.imshow(A2_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Oranges', alpha=0.5)
    ax2.axhline(y=0, color='black', linewidth=2,linestyle='--')
    ax2.plot(z,A2_center, color='red', linewidth=2,label='center position')
    ax2.legend(prop={'size': 16},loc='lower right')
    ax2.set_ylabel("x",fontsize=20)
    ax2.set_ylim(-10, 10)
    ax2.set_xticklabels([])  # Hide x-axis labels and ticks for A2
    hide_extreme_ytick_labels(ax2)

    add_title(ax2, title_text = r"$|A2|$")
    add_colorbar(ax2, cax2)
    adjust_axis_properties(ax2)
    add_index(ax2, text="(b)")
    ax2.tick_params(axis='both', labelsize=20)

    # Data3
    ax3 = plt.subplot(gs[2])

    cax3 = ax3.imshow(A3_abs,extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Greens', alpha=0.5)
    ax3.axhline(y=0, color='black', linewidth=2,linestyle='--')

    # Create the mask
    mask = (z > 21) & (z < 100)

    # Apply the mask
    z_masked = np.array(z)[mask]
    A3_center_masked = A3_center[mask]

    # Plot the masked values
    ax3.plot(z_masked, A3_center_masked, color='red', linewidth=2, label='center position')


    ax3.legend(prop={'size': 16},loc='lower right')
    ax3.set_ylabel("x",fontsize=20)
    ax3.set_ylim(-10, 10)
    ax3.set_xlabel("z",fontsize=20)  # Show x-axis labels and ticks only for A3
    hide_extreme_ytick_labels(ax3)
    add_title(ax3, title_text = r"$|A3|$")
    add_colorbar(ax3, cax3)
    adjust_axis_properties(ax3)
    add_index(ax3, text="(c)")
    ax3.tick_params(axis='both', labelsize=20)

    plt.show()

plot_center_inone(theta2 = 2,E1 = 0.1)
def plot_A2():
    A1_abs,A2_abs,A3_abs = load()
    # Plot the absolute value of A2 as a function of x and z
    plt.imshow(A2_abs, extent=[z.min(), z.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='jet')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.colorbar(label='|A2|')
    plt.title('Absolute value of A2 field')
    plt.show()

#plot_A2()

def power():
    A1_abs,A2_abs,A3_abs = load()

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

def reflection(theta2 = 2,E1 = 0.2):
    # Apply split-step Fourier method
    A1_abs,A2_abs,A3_abs = load(theta2 = theta2,E1 = E1,index = 5)



    # for array shape of (1000,50000)
    total_power = np.sum(np.square(A2_abs[649:1000, 4999:10000]))
    reflect_power = np.sum(np.square(A2_abs[649:1000, 31999:37000]))
    R = reflect_power/total_power
    print("reflection is ",R)
    return R

#reflection()

def reflection_alldata_get():
    E1_list = np.linspace(0.2, 0.6, 20)

    for  E1_value in E1_list :
        save_data(theta2 =  2.2, E1 = E1_value)

#reflection_alldata_get()

def reflection_alldata_analyze():
    E1_list = np.linspace(0.2, 0.6, 20)

    R1 =  []
    R2 =  []
    for  E1_value in E1_list :
        R1.append(reflection(theta2 =  2, E1 = E1_value))
    for  E1_value in E1_list :
        R2.append(reflection(theta2 =  2.2, E1 =  E1_value))
    plt.plot(E1_list,R1,label='theta2 =  2')
    plt.plot(E1_list,R2,label='theta2 =  2.2')
    print("R1 is ",R1)
    print("R2 is ",R2)
    plt.legend()
    plt.show()
    #R1 is  [0.18277668674356337, 0.19174758380417942, 0.20671423158032723, 0.2523866667531735, 0.3691838035643133, 0.5517523955818884, 0.7285780403538605, 0.8463380148268628, 0.9083859755010932, 0.9379067029114851, 0.9522075993899696, 0.9598560837713306, 0.9645478064516619, 0.9678679949487943, 0.9705019858137003, 0.9727415515946822, 0.9747116170054366, 0.9764805663416117, 0.9780852433579119, 0.9795507183193806]
    #R2 is  [0.09841458240555165, 0.10323026711002797, 0.11021252463169855, 0.14270290173621747, 0.2551534427604946, 0.4622554597761485, 0.680492608982278, 0.8288859319491764, 0.9049272616856316, 0.9392415444728685, 0.955221774019313, 0.9636534185548732, 0.9688024257398383, 0.9724421336746039, 0.9753107872450852, 0.9777163364637609, 0.9797988344542221, 0.9816386118575208, 0.9832809889127018, 0.9847586584994328]

#reflection_alldata_analyze()









