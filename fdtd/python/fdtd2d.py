#!/usr/bin/env python3
"""
2-D acoustic FDTD simulator (no staggering).

Usage:
    python fdtd2d.py config.json
"""
import json5 as json
import sys
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import time
from argparse import ArgumentParser

import numpy as np


def read_config(path):
    with open(path) as f:
        cfg = json.load(f)
    # sanity checks
    required = {
        "nx",
        "ny",
        "dx",
        "dy",
        "c",
        "rho",
        "n_steps",
        "output_every",
        "derivative",
        "source",
    }
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Config missing keys: {', '.join(sorted(missing))}")
    if "coeffs" not in cfg["derivative"] and "m" not in cfg["derivative"]:
        raise ValueError("derivative must contain either 'coeffs' or 'm'")
    return cfg

def make_fd_coeffs(m: int) -> np.ndarray:
    """
    Return the length-m vector [c1 … cm] such that

        df/dx ≈ Σ_{k=1..m} c_k (f_{i+k} - f_{i-k}) / h

    is 2m-th-order accurate on a uniform grid with spacing h.

    Parameters
    ----------
    m : int   -  half-width of the stencil (m ≥ 1)

    Examples
    --------
    >>> make_fd_coeffs(1)
    array([0.5])
    >>> make_fd_coeffs(4)
    array([ 0.8       , -0.2       ,  0.03809524, -0.00357143])
    """
    k = np.arange(1, m + 1, dtype=np.float32)          # 1 … m
    A = np.vstack([k ** (2*n + 1) for n in range(m)]).astype(np.float32)   # NO .T !
    rhs = np.zeros(m, dtype=np.float32)            # right-hand side
    rhs[0] = 0.5                                  # Σ c_k k = ½
    return np.linalg.solve(A, rhs)

# ------------------------------------------------------------------------- #
# central derivative with Neumann (mirror) edges
# ------------------------------------------------------------------------- #
def stencil_halo(field, coeffs, h, axis):
    """
    Central finite difference on a field that already carries a halo of
    width len(coeffs).  Returned array has *only the interior* values.
    """
    m = len(coeffs)
    f = field  # alias for brevity
    if axis == 0:  # y-derivative
        df = (
            sum(
                c_k * (f[m - k : -m - k, m:-m] - f[m + k : None if m == k else -m + k, m:-m])
                for k, c_k in enumerate(coeffs, 1)
            ) / h
        )
    else:  # x-derivative
        df = (
            sum(
                c_k * (f[m:-m, m - k : -m - k] - f[m:-m, m + k : None if m == k else -m + k])
                for k, c_k in enumerate(coeffs, 1)
            ) / h
        )
    return df


def divergence(vx, vy, coeffs, dx, dy):
    dvdx = stencil_halo(vx, coeffs, dx, axis=1)
    dvdy = stencil_halo(vy, coeffs, dy, axis=0)
    return dvdx + dvdy


def gradient(p, coeffs, dx, dy):
    """Return ∇p as (dp/dx, dp/dy)."""
    dpdx = stencil_halo(p, coeffs, dx, axis=1)
    dpdy = stencil_halo(p, coeffs, dy, axis=0)
    return dpdx, dpdy


def refresh_neumann(field, m):
    """Copy edge values outwards: ∂/∂n = 0 (rigid wall)."""
    # edges parallel to y
    field[:, :m] = field[:, m][:, None]  # west
    field[:, -m:] = field[:, -m - 1][:, None]  # east
    # edges parallel to x
    field[:m, :] = field[m, :][None, :]  # south
    field[-m:, :] = field[-m - 1, :][None, :]  # north


# -----------------------------------------------------------------------------#
# source wavelets
# -----------------------------------------------------------------------------#
def gaussian_wavelet(t, f0, amp=1.0):
    tau = 1.0 / f0
    t0 = 3 * tau
    return amp * np.exp(-((t - t0) ** 2) / (tau**2))


def ricker_wavelet(t, f0, amp=1.0):
    a = np.pi * f0 * (t - 1.0 / f0)  # centre near one period
    return amp * (1 - 2 * a**2) * np.exp(-(a**2))


def make_source_function(src_cfg):
    kind = src_cfg["type"].lower()
    f0 = src_cfg["frequency"]
    amp = src_cfg.get("amplitude", 1.0)
    if kind == "gaussian":
        return lambda t: gaussian_wavelet(t, f0, amp)
    elif kind == "ricker":
        return lambda t: ricker_wavelet(t, f0, amp)
    elif kind == "point":
        # point source is just a delta function at the source position
        return lambda t: amp if t == 0 else 0.0
    elif kind == "sin":
        # sinusoidal source, not very useful for FDTD
        return lambda t: amp * np.sin(2 * np.pi * f0 * t)
    else:
        raise ValueError(f"Unknown source type '{kind}'")


# -----------------------------------------------------------------------------#
# main time loop
# -----------------------------------------------------------------------------#
def run_sim(cfg, use_tqdm=False, output_file="wavefield.npz"):
    print("Running 2-D FDTD acoustic wave simulation with the following configuration:")
    pprint(cfg)
    nx, ny = cfg["nx"], cfg["ny"]
    dx, dy = cfg["dx"], cfg["dy"]
    c, rho = cfg["c"], cfg["rho"]
    n_steps, every = cfg["n_steps"], cfg["output_every"]
    deriv_cfg = cfg["derivative"]
    if "coeffs" in deriv_cfg:
        coeffs = np.asarray(deriv_cfg["coeffs"], dtype=np.float32)
    elif "m" in deriv_cfg:
        coeffs = make_fd_coeffs(int(deriv_cfg["m"]))
    else:
        raise ValueError("derivative needs either 'coeffs' or 'm'")
    m = len(coeffs)  # halo width = stencil half-size
    print(f"Using finite difference coefficients for m = {m}: {coeffs}")

    # --- stability (CFL) --- #
    cfl = 0.4  # fairly conservative for non-staggered update
    dt = cfl * min(dx, dy) / (c * np.sqrt(2.0))

    # --- fields --- #
    shape_full = (ny + 2 * m, nx + 2 * m)
    p = np.zeros(shape_full, dtype=np.float32)
    vx = np.zeros_like(p)
    vy = np.zeros_like(p)
    core = (slice(m, m + ny), slice(m, m + nx))  # interior slice

    # --- source --- #
    src_pos = tuple(
        m + np.array(cfg["source"]["position"][::-1])
    )  # (y,x) for numpy indexing
    source = make_source_function(cfg["source"])

    # --- storage --- #
    frames = None
    if every != 0:
        n_frames = (n_steps // every) + 1
        frames = np.zeros((n_frames, ny, nx), dtype=np.float32)
        frame_idx = 0
        frames[frame_idx] = p[core].copy()

    # --- time stepping --- #
    step_iter = tqdm(range(1, n_steps + 1), desc="Time steps", disable=not use_tqdm)
    sim_start = time.perf_counter()
    for it in step_iter:
        t = it * dt

        # update pressure
        div_v = divergence(vx, vy, coeffs, dx, dy)
        p[core] -= dt * rho * c**2 * div_v

        # add point source (pressure injection)
        p[src_pos] += source(t)

        # update velocity
        dpdx, dpdy = gradient(p, coeffs, dx, dy)
        vx[core] -= dt * dpdx / rho
        vy[core] -= dt * dpdy / rho

        # ---- rigid-wall (Neumann) condition: v_normal = 0  ------------------
        vx[:, m] = 0.0  # west
        vx[:, -m - 1] = 0.0  # east
        vy[m, :] = 0.0  # south
        vy[-m - 1, :] = 0.0  # north

        # ---- Refresh halos ----
        refresh_neumann(p, m)
        refresh_neumann(vx, m)
        refresh_neumann(vy, m)

        # record snapshot
        if every != 0 and it % every == 0:
            frame_idx += 1
            frames[frame_idx] = p[core].copy()
    sim_end = time.perf_counter()

    np.savez_compressed(output_file, frames=frames)
    print(f"Finished {n_steps} steps. (dt = {dt:.3e} s)")
    if every != 0:
        print(f"{frame_idx + 1} frames saved to wavefield.npz.")
    else:
        print("No frames saved (output_every = 0).")
    mptss = (ny * nx) * n_steps / (sim_end - sim_start) / 1e6  # million points per second
    print(f"Simulation took {sim_end - sim_start:.2f} seconds. Througput: {mptss:.2f} Mpts/s")


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python fdtd2d.py config.json")
    #     sys.exit(1)
    parser = ArgumentParser(description="2-D FDTD acoustic wave simulation")
    parser.add_argument("config", type=str, help="Path to the configuration JSON file")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")
    parser.add_argument("--output", type=str, default="wavefield.npz", help="Output file for wavefield data")
    args = parser.parse_args()
    if not Path(args.config).exists():
        print(f"Configuration file {args.config} does not exist.")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    cfg = read_config(config_path)
    run_sim(cfg, use_tqdm=args.tqdm, output_file=args.output)
