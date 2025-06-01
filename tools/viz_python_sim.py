import numpy as np, matplotlib.pyplot as plt, matplotlib.animation as anim
import argparse


def main():
    parser = argparse.ArgumentParser(description="2-D FDTD simulation visualization")
    parser.add_argument(
        "wavefield",
        type=str,
        help="Path to the wavefield data file (e.g., wavefield.npz)",
        default="wavefield.npz",
    )
    args = parser.parse_args()
    
    data = np.load(args.wavefield)["frames"]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0].T, cmap="seismic", origin="lower")
    
    def update(frame):
        im.set_data(data[frame].T)
        return (im,)

    # Run the animation
    ani = anim.FuncAnimation(fig, update, frames=len(data), interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()