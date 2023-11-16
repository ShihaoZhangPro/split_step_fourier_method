import numpy as np
import matplotlib.pyplot as plt

# Define constants
L = 30.0  # spatial domain (-L/2, L/2)
T = 2.0  # time domain (0, T)

N = 256  # number of spatial grid points
M = 256  # number of time steps

dx = L / N  # spatial grid size
dt = T / M  # time step size

x = np.arange(-L / 2, L / 2, dx)  # spatial grid
t = np.arange(0, T, dt)  # time grid

k = (2 * np.pi / L) * np.fft.fftfreq(N, d=dx)  # frequency grid

E2 = 1E-2
a2 = 1
k2 = 2
theta2 = 2
x0 = 8
# Initial condition
psi_0 = E2*np.exp(-(x - x0)**2/a2**2 + 1j*k2*theta2*x) #x0 = 8

# Solution matrix
psi = np.zeros((M, N), dtype=np.complex64)
psi[0, :] = psi_0

# Time evolution
for i in range(1, M):
    # Half time-step in real space
    psi_half = psi[i - 1, :] * np.exp(-0.5j * dt * np.abs(psi[i - 1, :]) ** 2)

    # Full time-step in Fourier space
    psi_hat = np.fft.fft(psi_half)
    psi_hat *= np.exp(-0.5j * dt * (k ** 2))

    # Back to real space
    psi_back = np.fft.ifft(psi_hat)

    # Another half time-step in real space
    psi[i, :] = psi_back * np.exp(-0.5j * dt * np.abs(psi_back) ** 2)

# Plot solution
plt.imshow(np.abs(psi) ** 2, extent=(-L / 2, L / 2, 0, T), aspect='auto', cmap='hot')
plt.colorbar(label='|psi|^2')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Solution of the Nonlinear Schrodinger Equation')
plt.show()
