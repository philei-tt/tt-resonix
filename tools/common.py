import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


def run_animation(frames, interval=30, clip=1.0, title="Title", save_path=None):
    frames -= frames.mean(axis=(1, 2), keepdims=True)
    ny, nx = frames.shape[1:]
    vmax = np.percentile(np.abs(frames), clip * 100)
    vmin = -vmax
    print(f"Using colour scale [{vmin:.3e}:{vmax:.3e}] Pa for {len(frames)} frames")

    fig, ax = plt.subplots()
    im = ax.imshow(
        frames[0].T,                            # transpose for (x,y) visual order
        origin="lower",
        cmap="seismic",
        vmin=vmin, 
        vmax=vmax,
        animated=True,                          # a tiny speed boost
    )
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")

    def update(i):
        im.set_data(frames[i].T)
        return (im,)

    a = anim.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

    if save_path is not None:
        print(f"Saving animation to {save_path} ...")
        a.save(save_path, writer='pillow', fps=1000 / interval)
        print("Animation saved.")

    plt.show()
