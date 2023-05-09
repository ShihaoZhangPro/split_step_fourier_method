
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dAdz(z, A, n, omega, gamma, theta2, x, x0, a1, a2):
    A1, A2, A3 = A
    n1, n2, n3 = n
    omega1, omega2, omega3 = omega
    gamma1, gamma2, gamma3 = gamma
    c = 3e8  # Speed of light in m/s

    k1 = n1 * omega1 / c
    k2 = n2 * omega2 / c
    k3 = n3 * omega3 / c
    k = np.array([k1, k2, k3])
    kz = k  # Assuming propagation along the z axis
    k_hat = kz[0] + kz[1] - kz[2]

    D = 1 / (2 * k)

    def Delta_perp(A):
        return 0

    dA1_dz = -1j * D[0] * Delta_perp(A1) - 1j * gamma1 * A3 * np.conj(A2) * np.exp(1j * k_hat * z)
    dA2_dz = -1j * D[1] * Delta_perp(A2) - 1j * gamma2 * A3 * np.conj(A1) * np.exp(1j * k_hat * z)
    dA3_dz = -1j * D[2] * Delta_perp(A3) - 1j * gamma3 * A1 * A2 * np.exp(-1j * k_hat * z)

    return np.array([dA1_dz, dA2_dz, dA3_dz])

def fun(z, A, *params):
    dAdz_val = dAdz(z, A, *params)
    return dAdz_val.T

# Define parameters and initial conditions
n = [1, 2, 3]
omega = [1e14, 2e14, 3e14]  # Some arbitrary frequencies
gamma = [1, 1, 1]
theta2 = 0.01
c = 3e8
n1, n2, n3 = n
omega1, omega2, omega3 = omega
k1 = n1 * omega1 / c
k2 = n2 * omega2 / c
k3 = n3 * omega3 / c

x = np.linspace(-10, 10, 200)
x0 = 1
a1 = 1
a2 = 1

E1 = 1
E2 = 0.1
A1_init = E1 * np.exp(-x**2 / a1**2)
A2_init = E2 * np.exp(-(x - x0)**2 / a2**2 + 1j * k2 * theta2 * x)
A3_init = 0

A_init = np.column_stack((A1_init, A2_init, A3_init))


# Define the range of x and z values
x_vals = np.linspace(-10, 10, 200)
z_vals = np.linspace(0, 10, 200)

# Solve the system of equations and evaluate the solutions for the given x and z values
A_vals = np.zeros((len(z_vals), len(x_vals), 3), dtype=complex)

for i, z in enumerate(z_vals):
    sol = solve_ivp(fun, (0, z), A_init, args=(n, omega, gamma, theta2, x, x0, a1, a2), method='RK45')
    A_vals[i] = sol.y[:, -1].reshape(-1, 3)

# Create the 2D plots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

c1 = ax[0].pcolormesh(x_vals, z_vals, np.abs(A_vals[:, :, 0]), cmap='viridis', shading='auto')
c2 = ax[1].pcolormesh(x_vals, z_vals, np.abs(A_vals[:, :, 1]), cmap='viridis', shading='auto')
c3 = ax[2].pcolormesh(x_vals, z_vals, np.abs(A_vals[:, :, 2]), cmap='viridis', shading='auto')

fig.colorbar(c1, ax=ax[0], label='|A1|')
fig.colorbar(c2, ax=ax[1], label='|A2|')
fig.colorbar(c3, ax=ax[2], label='|A3|')

ax[0].set_title('A1')
ax[1].set_title('A2')
ax[2].set_title('A3')

for a in ax:
    a.set_xlabel('x')
    a.set_ylabel('z')

plt.show()