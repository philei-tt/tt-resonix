import argparse
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

"""acoustic_fdtd.py ────────────────────────────────────────────────────────────
Staggered-grid FDTD (first-order, 2-D) — *fixed shape mismatch 2025-06-01*
───────────────────────────────────────────────────────────────────────────────
# What changed?
*   **Bug-fix** - shape mismatch when updating `vx`, `vy` (pressure gradient
    arrays were one index too wide).  We now slice off the last row/column so
    the shapes match the interior of the staggered field.

*   Added extra inline comments, especially in `central_diff`, to make the flow
    clearer for new readers.

See the big docstring of the 3-D file for usage; CLI is unchanged.
"""

# ────────────────────────── helper functions ────────────────────────────────

def ricker(f0: float, t: float) -> float:
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def gaussian(sigma: float, t: float) -> float:
    return np.exp(-0.5 * (t / sigma) ** 2)


# ───────────────── central-difference machinery ─────────────────────────────

def _central_kernel(coeffs: Sequence[float]) -> np.ndarray:
    """Given one-sided *coeffs* = [a₁, a₂, …], return full antisymmetric
    kernel  [ -a_m … -a₁ 0 +a₁ … +a_m ].  Length = 2m+1.
    """
    m = len(coeffs)
    k = np.zeros(2 * m + 1, dtype=np.float32)
    for i, a in enumerate(coeffs, 1):
        k[m + i] = +a   #  +i offset
        k[m - i] = -a   #  −i offset  (opposite sign)
    return k


def central_diff(arr: np.ndarray, axis: int, coeffs: Sequence[float], h: float) -> np.ndarray:
    """Return centred first derivative ∂arr/∂xₖ along *axis* using arbitrary-order
    symmetric coefficients.

    Steps (all NumPy, no SciPy):
    1.  **Pad** the array with `pad = m` zeros on both ends of the chosen axis so
        we can index the stencil even at the boundaries (hard-wall/Dirichlet).
    2.  **Re-orient** the axis of interest to the front (`moveaxis`) so the
        convolution becomes a pure 1-D problem.
    3.  Use `sliding_window_view` to create a *view* whose last dimension walks
        over every L-point window (L = length of kernel).
    4.  **Tensor-dot** the kernel with that window dimension → fast vectorised
        convolution without explicit loops.
    5.  **Move the axis back** and divide by grid spacing *h*.
    The output has *exactly the same shape* as `arr`.  Boundary rows/columns are
    automatically zero because of the padding.
    """
    kernel = _central_kernel(coeffs)          # step 0 – stencil coefficients
    pad = len(coeffs)                         # half-width m

    # 1. pad with zeros along *axis*
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    arr_p = np.pad(arr, pad_width, mode="constant", constant_values=0.0)

    # 2. bring the target axis to the front
    arr_p = np.moveaxis(arr_p, axis, 0)       # shape → (N+2m, ...)

    # 3. sliding window of length L = len(kernel)
    L = kernel.size
    sw = np.lib.stride_tricks.sliding_window_view(arr_p, window_shape=L, axis=0)

    # 4. tensordot kernel with window (collapses window axis)
    diff = np.tensordot(kernel, sw, axes=(0, -1))

    # 5. move axis back to original position and scale by h
    diff = np.moveaxis(diff, 0, axis)
    return diff / h


# ───────────────────────── simulator class ──────────────────────────────────

class AcousticFDTD:
    def __init__(self, cfg: dict):
        # grid geometry
        self.nx = int(cfg["nx"])
        self.ny = int(cfg["ny"])
        self.dx = float(cfg["dx"])
        self.dy = float(cfg["dy"])

        # medium properties
        self.c = float(cfg.get("c", 343.0))
        self.rho = float(cfg.get("rho", 1.225))

        # time step (CFL)
        if cfg.get("dt") is None:
            self.dt = 0.5 * min(self.dx, self.dy) / (self.c * np.sqrt(2))
        else:
            self.dt = float(cfg["dt"])
        self.n_steps = int(cfg.get("n_steps", 800))
        self.output_every = int(cfg.get("output_every", 20))

        # derivative coefficients
        self.coeffs = [float(x) for x in cfg.get("derivative", {}).get("coeffs", [0.5])]

        # staggered fields
        self.p = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.vx = np.zeros((self.nx + 1, self.ny), dtype=np.float32)
        self.vy = np.zeros((self.nx, self.ny + 1), dtype=np.float32)

        # source
        src = cfg.get("source", None)
        if src:
            self.sx, self.sy = map(int, src.get("position", [self.nx // 2, self.ny // 2]))
            self.amp = float(src.get("amplitude", 1.0))
            self.freq = float(src.get("frequency", 50.0))
            self.stype = src.get("type", "ricker").lower()
        else:
            self.sx = self.sy = None

        self.frames: List[np.ndarray] = []

    # ───────── low-level operations ──────────

    def _apply_source(self, n: int):
        if self.sx is None:
            return
        t = n * self.dt
        if self.stype == "ricker":
            val = self.amp * ricker(self.freq, t)
        else:
            val = self.amp * gaussian(1.0 / self.freq, t)
        self.p[self.sx, self.sy] += np.float32(val)

    def _update_velocity(self):
        # centred gradients at cell centres (same shape as p)
        dp_dx = central_diff(self.p, axis=0, coeffs=self.coeffs, h=self.dx)
        dp_dy = central_diff(self.p, axis=1, coeffs=self.coeffs, h=self.dy)
        print(f"[update_velocity] dp_dx shape: {dp_dx.shape}, dp_dy shape: {dp_dy.shape}, p shape: {self.p.shape}")

        # map to staggered faces by discarding the last row/col (now shapes match)
        self.vx[1:-1, :] -= (self.dt / self.rho) * dp_dx[:-1, :]  # (nx-1, ny)
        self.vy[:, 1:-1] -= (self.dt / self.rho) * dp_dy[:, :-1]  # (nx, ny-1)

        # hard-wall boundaries (v = 0)
        self.vx[0, :] = self.vx[-1, :] = 0.0
        self.vy[:, 0] = self.vy[:, -1] = 0.0
        print(f"[update_velocity] vx shape: {self.vx.shape}, vy shape: {self.vy.shape}")

    def _update_pressure(self):
        # dvx_dx = central_diff(self.vx, axis=0, coeffs=self.coeffs, h=self.dx)[1:-1, :]
        # dvy_dy = central_diff(self.vy, axis=1, coeffs=self.coeffs, h=self.dy)[:, 1:-1]
        dvx_dx_face = central_diff(self.vx, axis=0, coeffs=self.coeffs, h=self.dx)   # on faces
        dvx_dx = 0.5*(dvx_dx_face[1:,:] + dvx_dx_face[:-1,:])     # → centres

        dvy_dy_face = central_diff(self.vy, axis=1, coeffs=self.coeffs, h=self.dy)   # on faces
        dvy_dy = 0.5*(dvy_dy_face[:,1:] + dvy_dy_face[:,:-1])     # → centres

        print(f"[update_pressure] dvx_dx shape: {dvx_dx.shape}, dvy_dy shape: {dvy_dy.shape}")
        print(f"[update_pressure] p shape: {self.p.shape}")
        div = dvx_dx + dvy_dy
        self.p -= self.dt * self.rho * self.c ** 2 * div

        self.p[0, :] = self.p[-1, :] = 0.0
        self.p[:, 0] = self.p[:, -1] = 0.0

    # ───────── main loop ─────────

    def run(self, realtime: bool = False):
        if realtime:
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(self.p.T, cmap="seismic", origin="lower", vmin=-1e-3, vmax=1e-3)
            plt.title("pressure p")
            plt.pause(0.01)

        for n in range(self.n_steps):
            self._apply_source(n)
            self._update_velocity()
            self._update_pressure()

            if n % self.output_every == 0:
                self.frames.append(self.p.copy())
                if realtime:
                    im.set_data(self.p.T)
                    im.set_clim(vmin=self.p.min(), vmax=self.p.max())
                    plt.pause(0.001)

        if realtime:
            plt.show()

    def save(self, path="wavefield.npz"):
        np.savez_compressed(path, frames=np.stack(self.frames), dt=self.dt,
                            dx=self.dx, dy=self.dy, c=self.c, rho=self.rho)
        print(f"Saved {len(self.frames)} frames → {path}")


# ───────────────── CLI ─────────────────────


def main():
    ap = argparse.ArgumentParser(description="2-D staggered-grid acoustic FDTD")
    ap.add_argument("--config", required=True)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--output", default="wavefield.npz")
    args = ap.parse_args()

    cfg = json.load(open(Path(args.config)))
    sim = AcousticFDTD(cfg)
    print("Δt =", sim.dt)
    sim.run(realtime=args.show)
    sim.save(args.output)


if __name__ == "__main__":
    main()
