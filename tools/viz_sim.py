#!/usr/bin/env python3
"""
Visualise a 2-D FDTD pressure wavefield saved by fdtd2d.py.

Example
-------
    python viz_wavefield.py wavefield.npz --interval 20
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def main():
    parser = argparse.ArgumentParser(description="2-D FDTD simulation visualisation")
    parser.add_argument("wavefield", type=str, help="Path to the wavefield .npz file")
    parser.add_argument("--clip", type=float, metavar="P", default=1.0, help="Clip amplitude to P-percentile (0 < P < 1) (0.999 looks good)")
    parser.add_argument(
        "--interval", type=int, default=30, help="Milliseconds between animation frames"
    )
    args = parser.parse_args()

    path = Path(args.wavefield)
    if not path.exists():
        parser.error(f"{path} does not exist")

    frames = np.load(path)["frames"]            # shape (n_frames, ny, nx)
    frames -= frames.mean(axis=(1, 2), keepdims=True)
    ny, nx = frames.shape[1:]
    vmax = np.percentile(np.abs(frames), args.clip * 100)
    vmin = -vmax
    print(f"Using colour scale [{vmax:.3e}:{vmax:.3e}] Pa for {len(frames)} frames")

    fig, ax = plt.subplots()
    im = ax.imshow(
        frames[0].T,                            # transpose for (x,y) visual order
        origin="lower",
        cmap="seismic",
        vmin=vmin, 
        vmax=vmax,
        animated=True,                          # a tiny speed boost
        # interpolation="bilinear",  # smoother image
    )
    ax.set_title("Acoustic pressure field")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")

    def update(i):
        im.set_data(frames[i].T)
        return (im,)

    a = anim.FuncAnimation(fig, update, frames=len(frames), interval=args.interval, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
