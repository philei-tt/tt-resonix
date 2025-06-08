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
from common import run_animation


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

    # Use best (float64) precision for visualisation
    frames = np.load(path)["frames"].astype(np.float64)            # shape (n_frames, ny, nx)
    run_animation(
        frames, 
        interval=args.interval, 
        clip=args.clip, 
        title="Acoustic pressure field"
    )

if __name__ == "__main__":
    main()
