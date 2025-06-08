import argparse
from pathlib import Path

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

    # use best (float64) precision for comparison
    frames0 = np.load(path0)["frames"].astype(np.float64)            # shape (n_frames, ny, nx)
    frames1 = np.load(path1)["frames"].astype(np.float64)            # shape (n_frames, ny, nx)
    if frames0.shape != frames1.shape:
        parser.error(f"Wavefields have different shapes: {frames0.shape} vs {frames1.shape}")
    diff = np.abs(frames0 - frames1)
    max_diff = np.max(diff)
    print(f"Maximum difference between wavefields: {max_diff:.3e} Pa")
    
if __name__ == "__main__":
    main()
