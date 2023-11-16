import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mayavi import mlab

# Sample Data
data1 = np.random.rand(10, 500)
data2 = np.random.rand(10, 500)
data3 = np.random.rand(10, 500)

def plot_data(data1, data2, data3, show_ticks=False):
    fig = plt.figure(figsize=(15, 9))

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0)

    def add_colorbar(ax, cax_data):
        # Define custom position and size for colorbar
        left, bottom, width, height = ax.get_position().bounds
        cax = fig.add_axes([left + width + 0.003, bottom, 0.008, height / 2])
        fig.colorbar(cax_data, cax=cax)

    def add_title(ax, title_text):
        # Get the axis position and dimensions
        left, bottom, width, height = ax.get_position().bounds
        # Place title text at the right top corner of the axis
        ax.text(left + width + 0.03, bottom + height, title_text, 
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

    
    # Data1
    ax1 = plt.subplot(gs[0])
    cax1 = ax1.matshow(data1, cmap='viridis', aspect='auto')
    hide_extreme_ytick_labels(ax1)
    adjust_axis_properties(ax1)

    add_title(ax1, title_text = 'A1')
    add_colorbar(ax1, cax1)

    # Data2
    ax2 = plt.subplot(gs[1])
    cax2 = ax2.matshow(data2, cmap='viridis', aspect='auto')
    ax2.set_title('Data 2', y=1.1)
    add_colorbar(ax2, cax2)

    # Data3
    ax3 = plt.subplot(gs[2])
    cax3 = ax3.matshow(data3, cmap='viridis', aspect='auto')
    ax3.set_title('Data 3', y=1.1)
    add_colorbar(ax3, cax3)

    # Option to hide or show ticks
    if not show_ticks:
        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()


# Create a random dataset
data = np.random.rand(50, 50, 5000)

# Create a volume rendering of the data
src = mlab.pipeline.scalar_field(data)
volume = mlab.pipeline.volume(src)

# Set custom axis limits: x and y from 20 to -20, and z from 0 to 10
volume.actors.scale = [40/50, 40/50, 10/5000]
volume.actors.position = [-20, -20, 0]

# Add axes with labels and display values
axes = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', nb_labels=5, ranges=[20, -20, 20, -20, 0, 10])

mlab.show()