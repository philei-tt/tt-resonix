# make_density.py
import numpy as np

nx, ny = 300, 300
rho_bg = 1.225  # air
rho_blob = 10.0  # dense inclusion

x = np.arange(nx)[:, None]
y = np.arange(ny)[None, :]
cx, cy, r = 150, 150, 60  # centre & radius

mask = (x - cx) ** 2 + (y - cy) ** 2 < r**2
rho = np.full((nx, ny), rho_bg, dtype=np.float64)
rho[mask] = rho_blob

np.savetxt("rho_circle.txt", rho, fmt="%.6f")
print("wrote rho_circle.txt")
