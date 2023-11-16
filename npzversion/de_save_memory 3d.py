
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

def ssfm_once(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, k_x, dz):
    
    k_y = k_x
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
    FT2 = np.fft.fft2
    IFT2 = np.fft.ifft2

    A1 = IFT2(np.exp(1j * D1 * (k_x**2 + k_y**2) * dz * 0.5) * FT2(A1))
    A2 = IFT2(np.exp(1j * D2 * (k_x**2 + k_y**2) * dz * 0.5) * FT2(A2))
    A3 = IFT2(np.exp(1j * D3 * (k_x**2 + k_y**2) * dz * 0.5) * FT2(A3))



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


x = np.linspace(-20, 20, 512)
y = x
z = np.linspace( 0,35, 15000)
dz = z[1] - z[0]
dx = x[1] - x[0]
dy = dx
k_x = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
k_y = 2 * np.pi * np.fft.fftfreq(len(x), d=dy)


# Initial conditions
theta2 = 2
E1 = 0.4 # 0.2 0.6
E2 = 1e-2 #1e-2
a1 = 3 #3
a2 = 1 #1

z_size = 500
k_hat = -1*k1*k2*theta2*theta2/2.0/k3
print(k_hat)




def save_data(theta2 = 2,E1 = 0.2, a1 = 4,a2 = 1 ):
    E2 = 1e-2 #1e-2




    x0 = 8
    X, Y = np.meshgrid(x, y)  # create a 2D grid

    # A1
    A1_init = E1 * np.exp(-(X**2 + Y**2) / a1**2)

    # A2
    A2_init = E2 * np.exp(-((X - x0)**2 + Y**2) / a2**2 + 1j * k2 * theta2 * X)

    # A3
    A3_init = np.exp(-(X**2 + Y**2)) * 0



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
            f_name = f'3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}_{index}.npz'
            np.savez_compressed(f_name, **results)
            
            A1_results = []
            A2_results = []
            A3_results = []
        A1, A2, A3 = ssfm_once(D1, D2, D3, gamma1, gamma2, gamma3, k_hat, A1_init, A2_init, A3_init, k_x, dz)
        A1_results.append(np.abs(A1).astype(np.float32))
        A2_results.append(np.abs(A2).astype(np.float32))
        A3_results.append(np.abs(A3).astype(np.float32)) 
        A1_init, A2_init, A3_init = A1, A2, A3
    print(index)


   
save_data(theta2 = 2,E1 = 0.5, a1 = 4,a2 = 1)
save_data(theta2 = 2,E1 = 0.5, a1 = 2,a2 = 2)
save_data(theta2 = 2,E1 = 0.5, a1 = 1,a2 = 4)



def load(theta2 = 2,E1 = 0.2, a1 = 4,a2 = 1,index = 30):
    def fetch_data(f_name):
        dat = np.load(f_name)
        return dat['z'], dat['A1'],dat['A2'],dat['A3']
    



    f_name = f'3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}_{1}.npz'
    z,A1_results,A2_results,A3_results   = fetch_data(f_name)
    for i in range(2, index+1):
        f_name = f'3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}_{i}.npz'
        z,A1_array,A2_array,A3_array   = fetch_data(f_name)
        print(A1_results.shape)
        A1_results = np.concatenate((A1_results, A1_array), axis=0)
        A2_results = np.concatenate((A2_results, A2_array), axis=0)
        A3_results = np.concatenate((A3_results, A3_array), axis=0)




    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T

    print(A2_abs.shape)
    


    return A1_abs,A2_abs,A3_abs

#load(theta2 = 2,E1 = 0.5)

def extract(theta2 = 2,E1 = 0.2, a1 = 4,a2 = 1,index = 30):
    def fetch_data(f_name):
        dat = np.load(f_name)
        return dat['z'], dat['A1'],dat['A2'],dat['A3']
    



    f_name = f'3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}_{1}.npz'
    z,A1_results,A2_results,A3_results   = fetch_data(f_name)
    A1_results = A1_results[:, 255:256, :].squeeze(axis=1)
    A2_results = A2_results[:, 255:256, :].squeeze(axis=1)
    A3_results = A3_results[:, 255:256, :].squeeze(axis=1)
    for i in range(2, index+1):
        f_name = f'3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}_{i}.npz'
        z,A1_array,A2_array,A3_array   = fetch_data(f_name)
        print(A1_results.shape)
        A1_slice = A1_array[:, 255:256, :].squeeze(axis=1)
        A2_slice = A2_array[:, 255:256, :].squeeze(axis=1)
        A3_slice = A3_array[:, 255:256, :].squeeze(axis=1)
        A1_results = np.concatenate((A1_results, A1_slice), axis=0)
        A2_results = np.concatenate((A2_results, A2_slice), axis=0)
        A3_results = np.concatenate((A3_results, A3_slice), axis=0)

    A1_abs = np.abs(A1_results).T
    A2_abs = np.abs(A2_results).T
    A3_abs = np.abs(A3_results).T
    results = {
        "z"  : z,
        "A1" : np.array(A1_abs),
        "A2" : np.array(A2_abs),
        "A3" : np.array(A3_abs)
    }
    f_name = f'extract_3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}.npz'
    np.savez_compressed(f_name, **results)

#extract()
#extract(theta2 = 2,E1 = 0.5, a1 = 4,a2 = 1)
#extract(theta2 = 2,E1 = 0.5, a1 = 2,a2 = 2)
#extract(theta2 = 2,E1 = 0.5, a1 = 1,a2 = 4) 


def plot_all_extract(theta2 = 2,E1 = 0.2, a1 = 4,a2 = 1):
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

    def fetch_data(f_name):
        dat = np.load(f_name)
        return dat['z'], dat['A1'],dat['A2'],dat['A3']
    



    f_name = f'extract_3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}.npz'
    z,A1_abs,A2_abs,A3_abs   = fetch_data(f_name)


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
    mask = (z > 10) & (z < 31)

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


#plot_all_extract(theta2 = 2,E1 = 0.5, a1 = 1,a2 = 4)

def plot_xy(theta2 = 2,E1 = 0.5, a1 = 4,a2 = 1,index = 5,slice_index = 100,name = ""):

    def fetch_data(f_name):
        dat = np.load(f_name)
        return dat['z'], dat['A1'],dat['A2'],dat['A3']
    
    location = slice(slice_index, slice_index+1)
    f_name = f'3D_theta2={theta2}_E1={E1}_a1 ={a1}_a2 ={a2}_{index}.npz'
    z,A1_abs,A2_abs,A3_abs   = fetch_data(f_name)
    A1_slice = A1_abs[location, :, :].squeeze(axis=0)
    A2_slice = A2_abs[location, :, :].squeeze(axis=0)
    A3_slice = A3_abs[location, :, :].squeeze(axis=0)


    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot the absolute value of A1, A2, and A3 as a function of x and z
    ax.imshow(A1_slice, extent=[y.min(), y.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Blues', alpha=1)
    ax.imshow(A2_slice, extent=[y.min(), y.max(), x.min(), x.max()], aspect='auto', origin='lower', cmap='Reds', alpha=0.5)

    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Absolute values of A1, A2')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.5, label='|A1|'),
                       Patch(facecolor='red', alpha=0.5, label='|A2|'),
                       ]
    ax.legend(handles=legend_elements)
    
    fig.savefig(name)
    #plt.show()
    plt.close()


#plot_xy(theta2 = 2,E1 = 0.5, a1 = 4,a2 = 1,index = 5,slice_index = 100)

def save_frames(theta2 = 2,E1 = 0.5, a1 = 4,a2 = 1,index_number = 30,slice_number = 500):

    frame_index = 0
    for i in range(1,index_number):
        for j in np.linspace(start=0, stop=slice_number-1, num=5, dtype=int):

            plot_xy(theta2 = theta2,E1 = E1, a1 = a1,a2 =a2 ,index = i,slice_index = j,name = f"a1={a1}_a2={a2}_{frame_index:03d}.png")
            frame_index += 1 

    #ffmpeg -framerate 24 -i a1 =4_a2 =1_%03d.png -c:v libx264 -pix_fmt yuv420p out.mp4



save_frames(theta2 = 2,E1 = 0.5, a1 = 4,a2 = 1,index_number = 30,slice_number = 500)
save_frames(theta2 = 2,E1 = 0.5, a1 = 2,a2 = 2,index_number = 30,slice_number = 500)
save_frames(theta2 = 2,E1 = 0.5, a1 = 1,a2 = 4,index_number = 30,slice_number = 500)














