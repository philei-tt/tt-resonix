# make_obstacles.py
import numpy as np
nx, ny = 300, 300
mask = np.zeros((nx, ny), dtype=np.int8)

# rectangular reflective block
mask[140:160, 80:220] = 1

# # circular absorbing inclusion
# x = np.arange(nx)[:, None]
# y = np.arange(ny)[None, :]
# cx, cy, r = 75, 220, 40
# mask[(x-cx)**2 + (y-cy)**2 < r**2] = 2

np.savetxt("obstacles.txt", mask, fmt="%d")
print("wrote obstacles.txt")
