"""figure.py

Module conaining functions for plotting propagation dynamics.

Author: OM
Date: 2022-06-02
"""
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def fig_save(fig_name):
    if fig_name == None:
        plt.show()
    else:
        fig_format = os.path.splitext(fig_name)[-1]
        if fig_format == '.png':
            plt.savefig(fig_name, format='png', dpi=300)
            plt.close()
        elif fig_format == '.pdf':
            plt.savefig(fig_name, format='pdf', dpi=600)
            plt.close()
        elif fig_format == '.svg':
            plt.savefig(fig_name, format='svg', dpi=600)
            plt.close()
        else:
            plt.show()


def fig_subfig_label(fig, ax, label, loc=1):
        pos = ax.get_position()

        if loc==1:
            fig.text(pos.x0, pos.y1, label ,color='white',
                backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                boxstyle='square,pad=0.1'), verticalalignment='top' )

        else:
            print("check label position")
            exit()


def plot_intensities(x, z, A, o_name = None):

    # -- SET FIGURE STYLE -----------------------------------------------------
    fig_width = 3.5 # (inch)
    params = {
        'figure.figsize': (fig_width,fig_width*0.66),
        'legend.fontsize': 6,
        'legend.frameon': False,
        'axes.labelsize': 7,
        'axes.linewidth': 1.,
        'axes.linewidth': 0.8,
        'xtick.labelsize' :7,
        'ytick.labelsize': 7,
        'mathtext.fontset': 'stixsans',
        'mathtext.rm': 'serif',
        'mathtext.bf': 'serif:bold',
        'mathtext.it': 'serif:italic',
        'mathtext.sf': 'sans\\-serif',
        'font.size':  7,
        'font.family': 'serif',
        'font.serif': "Helvetica",
    }
    mpl.rcParams.update(params)


    # -- SET FIGURE LAYOUT ----------------------------------------------------
    f = plt.figure()
    plt.subplots_adjust(left = 0.12, bottom = 0.13, right = 0.98, top = 0.9, wspace = .5, hspace = 3.)

    gs00 = GridSpec(nrows = 1, ncols = 1)
    gsA = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs00[0,0], wspace=0.07, hspace=0.1)
    ax1 = f.add_subplot(gsA[0, 0])
    ax2 = f.add_subplot(gsA[0, 1])
    ax3 = f.add_subplot(gsA[0, 2])

    # -- SET AXES BOUNDS ------------------------------------------------------ 
    z_lim = (0,z[-1])
    #z_ticks = (0,20,40,60,80,100)
    x_lim = (-12,12)
    x_ticks = (-10,-5,0,5,10)

    # -- UNPACK FIELDS
    A1, A2, A3 = A

    def _setColorbar_v2(im, refPos, label = 'test'):
        """colorbar helper"""
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        dw = 0.5*w
        cax = f.add_axes([x0 + dw, y0 + 1.02 * h, w - dw, 0.03 * h])
        cbar = f.colorbar(im, cax=cax, orientation="horizontal", extend='max')
        cbar.ax.tick_params(
            color="k",
            labelcolor="k",
            bottom=False,
            direction="out",
            labelbottom=False,
            labeltop=True,
            top=True,
            size=2,
            pad = 1,
            length=2,
            labelsize = 5
        )

        f.text(x0, y0+1.02*h, label, horizontalalignment='left', verticalalignment='bottom', size = 6)
        return cbar

    # -- SUBFIGURE A ---------------------------------------------------------- 

    # ... NORMALIZE TO INITIAL MAXIMUM PEAK INITENSITY
    ax = ax1
    I1 = np.abs(A1) ** 2
    I1 /= np.max(I1)

    # ... PLOT INTENSITY
    cmap = mpl.cm.get_cmap("Greys")
    im1 = ax.pcolorfast(
        x,
        z,
        I1[:-1, :-1],
        norm=col.Normalize(vmin=0,vmax=1),
        cmap=cmap
        )
    cbar1 = _setColorbar_v2(im1, ax.get_position(), label= r"$|A_1(x,z)|^2$")
    cbar1.set_ticks((0,0.5,1))
    #cbar1.ax.set_title(r"$|A_1(x,z)|^2$", color="k", y=2., fontsize=6)
    ax.tick_params(axis='y', length=2., pad=2)
    ax.set_ylim(z_lim)
    #ax.set_yticks(z_ticks)
    ax.set_xlabel(r"Coordinate $x$")
    ax.tick_params(axis='x', length=2., pad=2, top=False)
    ax.set_xlim(x_lim)
    ax.set_xticks(x_ticks)
    ax.set_ylabel(r"Propagation distance $z$")

    fig_subfig_label(f, ax, r"(a)", loc=1)

    # ... GET CENTROID COORDINATE
    I = np.abs(A1)**2
    xc = np.sum(x[np.newaxis,:]*I,axis=-1)/np.sum(I,axis=-1)
    # ... ADD CENTROID COORDINATE
    #ax.plot(xc, z, color='white', dashes=[3,2], lw=0.75)

    # -- SUBFIGURE B ---------------------------------------------------------- 

    ax = ax2
    I2 = np.abs(A2) ** 2
    I2 /= np.max(I2)

    # ... PLOT INTENSITY
    cmap = mpl.cm.get_cmap("Blues")
    im1 = ax.pcolorfast(
        x,
        z,
        I2[:-1, :-1],
        norm=col.Normalize(vmin=0,vmax=1),
        cmap=cmap
        )
    cbar1 = _setColorbar_v2(im1, ax.get_position(), label= r"$|A_2(x,z)|^2$")
    cbar1.set_ticks((0,0.5,1))
    ax.tick_params(axis='y', length=2., pad=2, labelleft=False)
    ax.set_ylim(z_lim)
    #ax.set_yticks(z_ticks)
    ax.set_xlabel(r"Coordinate $x$")
    ax.tick_params(axis='x', length=2., pad=2, top=False)
    ax.set_xlim(x_lim)
    ax.set_xticks(x_ticks)
    #ax.set_ylabel(r"Propagation distance $z$")

    # ... ADD CENTROID COORDINATE
    ax.plot(xc, z, color='k', dashes=[3,2], lw=1)

    fig_subfig_label(f, ax, r"(b)", loc=1)

    # -- SUBFIGURE C ---------------------------------------------------------- 

    ax = ax3
    I = np.abs(A3) ** 2
    I /= np.max(I)

    # ... PLOT INTENSITY
    cmap = mpl.cm.get_cmap("Reds")
    im1 = ax.pcolorfast(
        x,
        z,
        I[:-1, :-1],
        norm=col.Normalize(vmin=0,vmax=1),
        cmap=cmap
        )
    cbar1 = _setColorbar_v2(im1, ax.get_position(), label= r"$|A_3(x,z)|^2$")
    cbar1.set_ticks((0,0.5,1))
    ax.tick_params(axis='y', length=2., pad=2, labelleft=False)
    ax.set_ylim(z_lim)
    #ax.set_yticks(z_ticks)
    ax.set_xlabel(r"Coordinate $x$")
    ax.tick_params(axis='x', length=2., pad=2, top=False)
    ax.set_xlim(x_lim)
    ax.set_xticks(x_ticks)
    #ax.set_ylabel(r"Propagation distance $z$")

    # ... ADD CENTROID COORDINATE
    ax.plot(xc, z, color='k', dashes=[3,2], lw=1)

    fig_subfig_label(f, ax, r"(c)", loc=1)

    # -- SAVE FIGURE
    fig_save(o_name)


def fetch_data(path):
    """fetch data

    Reads in data from file in numpy npz-format

    Args:
      path (str): path to npz-file

    Returns: (x, t, uxt)
      x (1D array): x samples
      t (1D array): t samples
      uxt (2D array): wave profile u(x,t)
    """
    dat = np.load(path)
    return dat['z'], dat['x'], dat['A1'], dat['A2'], dat['A3']


def main():
    f_name = 'C:/Users/Administrator/Desktop/Study/master project/split_step_fourier_method/EXAMPLE_SAVE_DATA/example_figure/res_3WQuMed_a13.000000_E11.000000_x0-8.000000_theta2-0.177778_a21.000000_E20.010000.npz'
    z, x, A1, A2, A3 = fetch_data(f_name)
    plot_intensities(x, z, (A1, A2, A3), o_name = './fig.png')


if __name__=="__main__":
    main()
