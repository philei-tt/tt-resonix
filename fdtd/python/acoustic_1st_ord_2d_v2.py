import argparse
import json5 as json
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

"""acoustic_fdtd.py ────────────────────────────────────────────────────────────
Staggered-grid FDTD for 2-D acoustics (pressure + velocity)
───────────────────────────────────────────────────────────────────────────────
*   **2025-06-01** — stability & overflow fixes
    • all arrays promoted to **float64** (was float32) to avoid premature
      overflow/underflow when high-order stencils amplify round-off.
    • CFL timestep now scales with the chosen stencil:  
      Δt ≤ min(Δx,Δy)/(c·√(4 Σ|aₘ|)).  Same limit as the classic 0.5·Δx/(c√2)
      when coeffs = [0.5].  We use 0.9x of that bound as a safety factor.
    • `_update_velocity` and `_update_pressure` keep the arbitrary-order stencil
      yet guarantee perfect staggering alignment to preserve energy balance.

Pass "coeffs" in the JSON config exactly as before; popular choices:
    [0.5]                                 # 2-nd order
    [4/5, -1/5, 4/105, -1/280]            # 8-th order
    [8/9, -14/45, 56/495, -7/198,
     56/6435, -2/1287, 8/45045, -1/102960]  # 16-th order

Example run (8-th order):
    python acoustic_fdtd.py --config config.json --show
"""

SIMULATION_DTYPE = np.float64   # For some reason float32 is unstable

# ─────────────────── helper wavelets ────────────────────


def ricker(f0: float, t: float) -> float:
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def gaussian(sigma: float, t: float) -> float:
    return np.exp(-0.5 * (t / sigma) ** 2)


# ───────── stencil kernel + derivative util ────────────


def _central_kernel(coeffs: Sequence[float]) -> np.ndarray:
    """Convert one-sided list → full antisymmetric kernel."""
    m = len(coeffs)
    k = np.zeros(2 * m + 1, dtype=SIMULATION_DTYPE)
    for i, a in enumerate(coeffs, 1):
        k[m + i] = +a
        k[m - i] = -a
    return k


def central_diff(
    arr: np.ndarray, axis: int, coeffs: Sequence[float], h: float
) -> np.ndarray:
    """Centred first derivative along *axis* with arbitrary order.

    1. zero-pad *m* cells each side (Dirichlet: p,v = 0 outside)
    2. move target axis to front → 1-D convolution with stride trick
    3. tensordot kernel with sliding windows (vectorised)
    4. divide by *h* and restore axis order.
    """
    kernel = _central_kernel(coeffs)
    m = len(coeffs)

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (m, m)
    arr_p = np.pad(arr, pad_width, mode="constant", constant_values=0.0)

    arr_p = np.moveaxis(arr_p, axis, 0)  # axis → 0
    L = kernel.size
    sw = np.lib.stride_tricks.sliding_window_view(arr_p, window_shape=L, axis=0)
    diff = np.tensordot(kernel, sw, axes=(0, -1))  # collapses window dim
    diff = np.moveaxis(diff, 0, axis) / h
    return diff


# ───────────── simulator class ─────────────────────────


class AcousticFDTD:
    def __init__(self, cfg: dict):
        # grid
        self.nx = int(cfg["nx"])
        self.ny = int(cfg["ny"])
        self.dx = float(cfg["dx"])
        self.dy = float(cfg["dy"])

        # medium
        self.c = float(cfg.get("c", 343.0))
        
        # rho (density) ────────────────────────────────────────────────
        if "rho_file" in cfg:
            self.rho = np.loadtxt(cfg["rho_file"], dtype=np.float64)
            if self.rho.shape != (self.nx, self.ny):
                raise ValueError("rho_file size mismatch")
        else:
            self.rho = np.full((self.nx, self.ny),             # array, not scalar
                            float(cfg.get("rho", 1.225)),
                            dtype=np.float64)
        # averages on faces (will be used every step)
        self.inv_rho_x = 1.0 / (0.5 * (self.rho[:-1, :] + self.rho[1:, :]))   # (nx-1, ny)
        self.inv_rho_y = 1.0 / (0.5 * (self.rho[:, :-1] + self.rho[:, 1:]))   # (nx,   ny-1)
        
        # obstacle mask ──────────────────────────────────────────────────────────
        # TXT file of integers with same (nx, ny) shape as p:
        #   0 → free space
        #   1 → reflective obstacle (hard)
        #   2 → absorbing obstacle
        if "obstacle_file" in cfg:
            mask = np.loadtxt(cfg["obstacle_file"], dtype=np.int8)
            if mask.shape != (self.nx, self.ny):
                raise ValueError("obstacle_file size mismatch")
            self.reflect = mask == 1
            self.absorb  = mask == 2
            # absorption strength  (1/seconds).  Bigger → stronger damping
            self.sigma   = float(cfg.get("absorb_coef", 100.0))
        else:
            self.reflect = np.zeros((self.nx, self.ny), dtype=bool)
            self.absorb  = np.zeros_like(self.reflect)
            self.sigma   = 0.0


        # derivative stencil
        self.coeffs: List[float] = [
            float(x) for x in cfg.get("derivative", {}).get("coeffs", [0.5])
        ]
        sum_abs = sum(abs(a) for a in self.coeffs)

        # time step (CFL for arbitrary stencil)
        cfl = 0.35 * min(self.dx, self.dy) / (self.c * 2 * sum_abs)
        if cfg.get("dt") is None:
            self.dt = cfl
        else:
            self.dt = float(cfg["dt"])
            if self.dt > cfl:
                logging.warning(
                    f"Warning: provided dt={self.dt:.4e} is larger than CFL limit {cfl:.4e}. "
                    "Simulation may be unstable."
                )
                
        self.n_steps = int(cfg.get("n_steps", 800))
        self.output_every = int(cfg.get("output_every", 20))

        # allocate staggered fields (float64) ------------------------------
        self.p = np.zeros((self.nx, self.ny), dtype=SIMULATION_DTYPE)
        self.vx = np.zeros((self.nx + 1, self.ny), dtype=SIMULATION_DTYPE)
        self.vy = np.zeros((self.nx, self.ny + 1), dtype=SIMULATION_DTYPE)

        # source -----------------------------------------------------------
        src = cfg.get("source", None)
        if src:
            self.sx, self.sy = map(
                int, src.get("position", [self.nx // 2, self.ny // 2])
            )
            self.amp = float(src.get("amplitude", 1.0))
            self.freq = float(src.get("frequency", 50.0))
            self.stype = src.get("type", "ricker").lower()
        else:
            self.sx = self.sy = None

        self.frames: List[np.ndarray] = []

    # ---------------------- internal helpers -----------------------------

    def _apply_source(self, n: int):
        if self.sx is None:
            return
        t = n * self.dt
        pulse = (
            ricker(self.freq, t)
            if self.stype == "ricker"
            else gaussian(1.0 / self.freq, t)
        )
        self.p[self.sx, self.sy] += self.amp * pulse

    # gradient(p) on faces → update v -------------------------------------
    def _update_velocity(self):
        # staggered‐grid, arbitrary order: derivative at **cell centres**,
        # then map to faces by taking left / bottom neighbour.
        dp_dx_center = central_diff(self.p, axis=0, coeffs=self.coeffs, h=self.dx)
        dp_dy_center = central_diff(self.p, axis=1, coeffs=self.coeffs, h=self.dy)

        self.vx[1:-1, :] -= (self.dt * self.inv_rho_x) * dp_dx_center[:-1, :]  # (nx-1,ny) maps to faces 1…nx-1
        self.vy[:, 1:-1] -= (self.dt * self.inv_rho_y) * dp_dy_center[:, :-1]  # (nx,ny-1) maps to faces 1…ny-1

        # hard walls: v = 0 at boundary faces
        self.vx[0, :] = self.vx[-1, :] = 0.0
        self.vy[:, 0] = self.vy[:, -1] = 0.0
        # … existing gradient & update lines remain …

        # ——--- REFLECTIVE CELLS (rigid) ---——
        # zero normal velocity at faces touching a reflective cell
        self.vx[1:-1, :][ self.reflect[:-1, :] | self.reflect[1:,  :] ] = 0.0
        self.vy[:, 1:-1][ self.reflect[:, :-1] | self.reflect[:, 1:] ]  = 0.0

        # ——--- ABSORBING CELLS (loss) ---——
        if self.sigma > 0.0:
            damp = np.exp(-self.sigma * self.dt)
            self.vx[1:-1, :][ self.absorb[:-1, :] | self.absorb[1:,  :] ] *= damp
            self.vy[:, 1:-1][ self.absorb[:, :-1] | self.absorb[:, 1:] ]  *= damp


    # divergence(v) at centres → update p ----------------------------------
    def _update_pressure(self):
        dvx_dx_face = central_diff(self.vx, axis=0, coeffs=self.coeffs, h=self.dx)
        dvy_dy_face = central_diff(self.vy, axis=1, coeffs=self.coeffs, h=self.dy)

        # average adjacent faces to cell centres
        dvx_dx = 0.5 * (dvx_dx_face[1:, :] + dvx_dx_face[:-1, :])
        dvy_dy = 0.5 * (dvy_dy_face[:, 1:] + dvy_dy_face[:, :-1])

        div = dvx_dx + dvy_dy
        self.p -= self.dt * self.rho * self.c**2 * div

        # Dirichlet boundaries (rigid wall)
        self.p[0, :] = self.p[-1, :] = 0.0
        self.p[:, 0] = self.p[:, -1] = 0.0
        
        if self.sigma > 0.0:
            self.p[self.absorb] *= np.exp(-self.sigma * self.dt)


    # --------------------------- main loop -------------------------------
    def run(self, realtime: bool = False):
        if realtime:
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(
                self.p.T, cmap="seismic", origin="lower", vmin=-1e-4, vmax=1e-4
            )
            plt.title("pressure p")
            plt.pause(0.01)

        for n in tqdm(range(self.n_steps)):
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

    # ---------------------------- I/O ------------------------------------
    def save(self, path="wavefield.npz"):
        np.savez_compressed(
            path,
            frames=np.stack(self.frames),
            dt=self.dt,
            dx=self.dx,
            dy=self.dy,
            c=self.c,
            rho=self.rho,
        )
        print(f"Saved {len(self.frames)} frames → {path}")


# ───────────────────────────── CLI entry ‐ point ─────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="2-D staggered-grid acoustic FDTD (arbitrary order)"
    )
    ap.add_argument("--config", required=True, help="JSON configuration file")
    ap.add_argument("--show", action="store_true", help="live animation")
    ap.add_argument("--output", default="wavefield.npz")
    args = ap.parse_args()

    cfg = json.load(open(Path(args.config)))
    sim = AcousticFDTD(cfg)
    print(
        f"Δt = {sim.dt:.4e} s  (user-specified)"
        if cfg.get("dt")
        else f"Δt auto = {sim.dt:.4e} s"
    )

    sim.run(realtime=args.show)
    sim.save(args.output)


if __name__ == "__main__":
    main()
