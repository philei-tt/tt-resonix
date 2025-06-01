import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import logging

"""Finite-difference time-domain (FDTD) simulation of 2-D acoustic wave propagation.

Usage (minimal):
    python acoustic_fdtd.py --config config.json [--initial initial_state.txt] [--show]

The simulation parameters are provided in a JSON file (example below).  Optionally, an
initial pressure field can be loaded from a plain-text file.  Frames of the pressure
field are stored in a NumPy “npz” file; with --show the wavefield is also displayed
in real time using matplotlib.

Example ``config.json``
----------------------
{
  "nx": 200,
  "ny": 200,
  "dx": 0.01,
  "dy": 0.01,
  "c": 343.0,
  "dt": null,
  "n_steps": 1000,
  "output_every": 5,
  "source": {
    "type": "ricker",
    "frequency": 50.0,
    "position": [100, 100],
    "amplitude": 1.0
  },
  "boundary": "dirichlet"
}

``initial_state.txt`` (optional) is an ASCII grid of ``nx``x``ny`` numbers.
"""

# ---------------------------------- helpers ----------------------------------


def ricker_wavelet(f, t):
    """Ricker (Mexican-hat) wavelet of peak frequency *f* at time *t* (numpy array)."""
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def gaussian_pulse(sigma, t):
    """Simple Gaussian pulse of width *sigma* centred at t=0."""
    return np.exp(-0.5 * (t / sigma) ** 2)


# -------------------------------- simulation ---------------------------------


class AcousticFDTD:
    """2-D acoustic wave simulator using second-order FDTD."""

    def __init__(self, cfg, initial_path=None):
        # Grid geometry
        self.nx = int(cfg["nx"])
        self.ny = int(cfg["ny"])
        self.dx = float(cfg["dx"])
        self.dy = float(cfg["dy"])
        self.c = float(cfg.get("c", 343.0))  # speed of sound (m/s)

        # Time stepping
        cfl_dt = 0.5 * min(self.dx, self.dy) / (self.c * np.sqrt(2))
        if cfg.get("dt") is None:
            # CFL: dt <= 1 / (c * sqrt((1/dx^2)+(1/dy^2)))
            self.dt = cfl_dt
        else:
            self.dt = float(cfg["dt"])
            if self.dt > cfl_dt:
                logging.warning(
                    f"Provided dt={self.dt} is larger than CFL limit {cfl_dt:.4f}. "
                    "Simulation may be unstable."
                )
        self.n_steps = int(cfg.get("n_steps", 1000))
        self.output_every = int(cfg.get("output_every", 10))

        # Source definition
        self.source = cfg.get("source", None)
        if self.source is not None:
            sx, sy = self.source.get("position", [self.nx // 2, self.ny // 2])
            self.sx = int(sx)
            self.sy = int(sy)
            self.samp = float(self.source.get("amplitude", 1.0))
            self.sfreq = float(self.source.get("frequency", 50.0))
            self.stype = self.source.get("type", "ricker").lower()
        else:
            self.sx = self.sy = None

        # Allocate pressure fields (previous, current, next)
        self.prev = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.curr = np.zeros_like(self.prev)
        self.next = np.zeros_like(self.prev)

        # Optional initial condition
        if initial_path is not None:
            arr = np.loadtxt(initial_path, dtype=np.float32)
            if arr.shape != self.curr.shape:
                raise ValueError("initial_state size mismatch")
            self.curr[:] = arr
            self.prev[:] = arr  # zero initial velocity

        # Coefficient for Laplacian term
        self.coeff = (self.c * self.dt) ** 2

        # Storage for outputs
        self.saved_frames = []

    # --------------------------- low-level operations ------------------------

    def _apply_source(self, n):
        if self.source is None:
            return
        t = n * self.dt
        if self.stype == "ricker":
            val = self.samp * ricker_wavelet(self.sfreq, t)
        elif self.stype == "gaussian":
            val = self.samp * gaussian_pulse(1.0 / self.sfreq, t)
        else:
            raise ValueError("Unknown source type")
        self.curr[self.sx, self.sy] += val

    def _step(self, n):
        """Perform one FDTD update."""
        # Interior Laplacian (vectorised, avoiding Python loops)
        lap = (
            self.curr[2:, 1:-1] - 2 * self.curr[1:-1, 1:-1] + self.curr[:-2, 1:-1]
        ) / self.dx**2 + (
            self.curr[1:-1, 2:] - 2 * self.curr[1:-1, 1:-1] + self.curr[1:-1, :-2]
        ) / self.dy**2
        self.next[1:-1, 1:-1] = (
            2 * self.curr[1:-1, 1:-1] - self.prev[1:-1, 1:-1] + self.coeff * lap
        )

        # Boundary: simple Dirichlet (fixed p=0)
        self.next[0, :] = 0
        self.next[-1, :] = 0
        self.next[:, 0] = 0
        self.next[:, -1] = 0

        # Rotate arrays
        self.prev, self.curr, self.next = self.curr, self.next, self.prev

    def run(self, realtime=False):
        if realtime:
            fig, ax = plt.subplots()
            im = ax.imshow(
                self.curr.T,
                cmap="seismic",
                vmin=-1,
                vmax=1,
                origin="lower",
                interpolation="nearest",
            )
            plt.title("Acoustic wavefield (pressure)")
            plt.pause(0.01)

        for n in range(self.n_steps):
            self._apply_source(n)
            self._step(n)

            if n % self.output_every == 0:
                self.saved_frames.append(self.curr.copy())
                if realtime:
                    im.set_data(self.curr.T)
                    im.set_clim(vmin=self.curr.min(), vmax=self.curr.max())
                    plt.pause(0.001)

        if realtime:
            plt.show()

    def save(self, out_path="wavefield.npz"):
        """Save recorded frames to a NumPy .npz file."""
        np.savez_compressed(out_path, frames=np.stack(self.saved_frames), dt=self.dt)


# --------------------------------- CLI glue ----------------------------------


def main():
    parser = argparse.ArgumentParser(description="2-D acoustic FDTD simulation")
    parser.add_argument(
        "--config", required=True, help="Path to JSON configuration file"
    )
    parser.add_argument("--initial", help="Optional initial-state text file")
    parser.add_argument(
        "--show", action="store_true", help="Display wavefield in real time"
    )
    parser.add_argument(
        "--output", default="wavefield.npz", help="Output .npz file for saved frames"
    )
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    sim = AcousticFDTD(cfg, initial_path=args.initial)
    sim.run(realtime=args.show)
    sim.save(args.output)
    print(f"Simulation finished. Frames saved to {args.output}")


if __name__ == "__main__":
    main()
