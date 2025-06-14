import argparse
from pathlib import Path
from common import run_animation

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="2-D FDTD simulation visualisation")
    parser.add_argument("wavefield0", type=str, help="Path to the wavefield .npz file")
    parser.add_argument("wavefield1", type=str, help="Path to the wavefield .npz file")
    args = parser.parse_args()

    path0 = Path(args.wavefield0)
    path1 = Path(args.wavefield1)
    if not path0.exists() or not path1.exists():
        parser.error(f"{path0} or {path1} does not exist")

    # Use best (float64) precision for visualisation
    frames0 = np.load(path0)["frames"].astype(np.float64)            # shape (n_frames, ny, nx)
    frames1 = np.load(path1)["frames"].astype(np.float64)            # shape (n_frames, ny, nx)
    if frames0.shape != frames1.shape:
        parser.error(f"Wavefields have different shapes: {frames0.shape} vs {frames1.shape}")
    diff = np.abs(frames0 - frames1)
    run_animation(
        diff, 
        interval=30, 
        clip=1.0, 
        title="Difference between wavefields"
    )
    
if __name__ == "__main__":
    main()
