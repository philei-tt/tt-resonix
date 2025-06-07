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
    if "coeffs" not in cfg["derivative"]:
        raise ValueError("derivative must contain a 'coeffs' array")
    return cfg


# ------------------------------------------------------------------------- #
# central derivative with Neumann (mirror) edges
# ------------------------------------------------------------------------- #
def stencil(field, coeffs, h, axis):
    """
    Central finite difference with an arbitrary symmetric stencil and
    zero-gradient (Neumann) boundaries obtained by edge mirroring.

    Parameters
    ----------
    field : 2-D ndarray
    coeffs : sequence of length m   - weights c₁ … c_m
    h : float                       - grid spacing in chosen axis
    axis : 0 for y-direction, 1 for x-direction
    """
    m = len(coeffs)  # stencil half-width
    # pad with mirrored edge values so that ∂/∂n = 0 at the wall
    f_ext = np.pad(field, ((m, m), (m, m)), mode="edge")

    slc_core = (
        slice(m, m + field.shape[0]),
        slice(m, m + field.shape[1]),
    )
    df = np.zeros_like(field)

    for k, c_k in enumerate(coeffs, start=1):
        if axis == 0:  # derivative along y
            up = slice(m - k, m - k + field.shape[0])
            down = slice(m + k, m + k + field.shape[0])
            df += c_k * (f_ext[up, slc_core[1]] - f_ext[down, slc_core[1]])
        else:  # derivative along x
            left = slice(m - k, m - k + field.shape[1])
            right = slice(m + k, m + k + field.shape[1])
            df += c_k * (f_ext[slc_core[0], left] - f_ext[slc_core[0], right])

    return df / h


def divergence(vx, vy, coeffs, dx, dy):
    dvdx = stencil(vx, coeffs, dx, axis=1)
    dvdy = stencil(vy, coeffs, dy, axis=0)
    return dvdx + dvdy


def gradient(p, coeffs, dx, dy):
    """Return ∇p as (dp/dx, dp/dy)."""
    dpdx = stencil(p, coeffs, dx, axis=1)
    dpdy = stencil(p, coeffs, dy, axis=0)
    return dpdx, dpdy


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
def run_sim(cfg):
    print("Running 2-D FDTD acoustic wave simulation with the following configuration:")
    pprint(cfg)
    nx, ny = cfg["nx"], cfg["ny"]
    dx, dy = cfg["dx"], cfg["dy"]
    c, rho = cfg["c"], cfg["rho"]
    n_steps, every = cfg["n_steps"], cfg["output_every"]
    coeffs = cfg["derivative"]["coeffs"]

    # --- stability (CFL) --- #
    cfl = 0.4  # fairly conservative for non-staggered update
    dt = cfl * min(dx, dy) / (c * np.sqrt(2.0))

    # --- fields --- #
    p = np.zeros((ny, nx), dtype=np.float32)
    vx = np.zeros_like(p)
    vy = np.zeros_like(p)

    # --- source --- #
    src_pos = tuple(cfg["source"]["position"][::-1])  # (y,x) for numpy indexing
    source = make_source_function(cfg["source"])

    # --- storage --- #
    n_frames = (n_steps // every) + 1
    frames = np.zeros((n_frames, ny, nx), dtype=np.float32)
    frame_idx = 0
    frames[frame_idx] = p.copy()

    # --- time stepping --- #
    for it in tqdm(range(1, n_steps + 1)):
        t = it * dt

        # update pressure
        div_v = divergence(vx, vy, coeffs, dx, dy)
        p -= dt * rho * c**2 * div_v

        # add point source (pressure injection)
        p[src_pos] += source(t)

        # update velocity
        dpdx, dpdy = gradient(p, coeffs, dx, dy)
        vx -= dt * dpdx / rho
        vy -= dt * dpdy / rho

        # ---- Neumann (rigid) boundary ----
        vx[:, 0] = 0.0  # west wall   (normal = x-direction)
        vx[:, -1] = 0.0  # east wall
        vy[0, :] = 0.0  # south wall  (normal = y-direction)
        vy[-1, :] = 0.0  # north wall

        # record snapshot
        if it % every == 0:
            frame_idx += 1
            frames[frame_idx] = p.copy()

    np.savez_compressed("wavefield.npz", frames=frames)
    print(
        f"Finished {n_steps} steps. "
        f"{frame_idx + 1} frames saved to wavefield.npz (dt = {dt:.3e} s)."
    )


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fdtd2d.py config.json")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    cfg = read_config(config_path)
    run_sim(cfg)
