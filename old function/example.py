import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure, the axis, and the plot elements
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-')

def init():
    # Set the limits of the plot
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    # Update the data of the plot for each frame
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                   init_func=init, blit=True)

# Display the animation
plt.show()
ani.save('sine_wave_animation.mp4', writer='ffmpeg', fps=30)