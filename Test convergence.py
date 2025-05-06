import math
import numpy as np
import matplotlib.pyplot as plt

# Function to compute wave magnitude for a given grid size
def compute_wave_magnitude(nmax, jmax):
    x1, x2, y2 = -0.5, 2.1, 1.0
    c = 1.0  # Advection speed

    dx = (x2 - x1) / (nmax - 1)
    dy = y2 / (jmax - 1)

    x = np.zeros((nmax, jmax))
    y = np.zeros((nmax, jmax))
    R = np.zeros((nmax, jmax))

    # Mesh coordinates
    for j in range(jmax):
        x[:, j] = np.linspace(x1, x2, nmax)
    for n in range(nmax):
        y[n, :] = np.linspace(0, y2, jmax)

    # March explicitly in x, solving for the unknown Riemann invariant, R
    for n in range(nmax - 1):
        # Apply boundary condition at y=0
        R[n + 1, 0] = math.exp(-20 * (x[n + 1, 0] - 0.2) ** 2)

        # Update interior values using a first-order accurate upwind scheme
        for j in range(1, jmax):
            R[n + 1, j] = R[n, j] - c * dx / dy * (R[n, j] - R[n, j - 1])

    # Compute wave magnitude at upper boundary
    wavemag = 0.0
    for n in range(nmax - 1):
        avg = 0.5 * (R[n, jmax - 1] + R[n + 1, jmax - 1])
        wavemag += avg * dx

    return wavemag

# Grid sizes to test
grid_sizes = [(40, 15),(40, 16), (40, 17), (40, 18), (40, 19), (40, 20), (40, 21)]
wave_magnitudes = []

# Compute wave magnitude for each grid size
for nmax, jmax in grid_sizes:
    wavemag = compute_wave_magnitude(nmax, jmax)
    wave_magnitudes.append(wavemag)
    print(f"Grid size (nmax, jmax): ({nmax}, {jmax}), Wave magnitude: {wavemag}")

# Plot wave magnitude versus jmax
jmax_values = [size[1] for size in grid_sizes]
plt.figure(figsize=(10, 6))
#plt.ylim(0.3, 0.5)
plt.plot(jmax_values, wave_magnitudes, marker='o', linestyle='-', color='r')
plt.xlabel('jmax (Number of mesh points in j)')
plt.ylabel('Wave Magnitude')
plt.title('Wave Magnitude vs jmax')
plt.grid(True)
plt.show()