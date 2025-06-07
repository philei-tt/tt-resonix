# function_plotter.py

import matplotlib.pyplot as plt
import numpy as np

def visualize_function(f, x_start, x_end, num_points=1000):
    """
    Visualize a function y = f(x) on the range [x_start, x_end].

    Parameters:
        f (callable): the function to plot.
        x_start (float): start of the x range.
        x_end (float): end of the x range.
        num_points (int): number of points to sample in the range.
    """
    # Generate x values
    x = np.linspace(x_start, x_end, num_points)
    # Compute y values
    y = f(x)

    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'f(x)', linewidth=2)
    plt.title('Function Visualization')
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def ricker_wavelet(t, f0, amp=1.0):
    a = np.pi * f0 * (t - 1.0 / f0)  # centre near one period
    return amp * (1 - 2 * a**2) * np.exp(-(a**2))

def gaussian_wavelet(t, f0, amp=1.0):
    tau = 1.0 / f0
    t0 = 3 * tau
    return amp * np.exp(-((t - t0) ** 2) / (tau**2))

if __name__ == '__main__':
    # Example usage: plot y = sin(x)
    import math

    visualize_function(lambda x: gaussian_wavelet(x, 80, 1), x_start=0, x_end=1.649e-06 * 1500 * np.pi)
