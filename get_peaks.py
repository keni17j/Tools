"""Get peaks with two ways."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def main():
    """Main function."""

    # Make a sample data.
    x = np.arange(-10, 10, 0.5)
    y = -(x-3) ** 2 + 100
    noise = np.random.randint(0, 20, y.shape[0])
    y = y + noise

    # Get a peak.
    peak_x, peak_y = peak_polyfit(x, y)
    print('peak_x', peak_x)
    print('peak_y', peak_y)

    # Get peaks.
    peaks_x, peaks_y = peak_find_peaks(x, y)
    print('peaks_x', peaks_x)
    print('peaks_y', peaks_y)

    # Show a graph.
    draw_graph(x, y,
               peak_x, peak_y,
               peaks_x, peaks_y,
               )


def peak_polyfit(x, y):
    """Get a peak using np.polyfit.
    The approximate function is quadratic function.
    """

    # y = ax^2 + bx + c.
    coe = np.polyfit(x, y, 2)
    a = coe[0]
    b = coe[1]
    c = coe[2]
    peak_x = -b / (2*a)
    peak_y = a * peak_x**2 + b * peak_x + c

    return peak_x, peak_y


def peak_find_peaks(x, y):
    """Get a peak using scipy.signal.find_pekas."""

    peaks, _ = signal.find_peaks(y)
    peaks_x = x[peaks]
    peaks_y = y[peaks]

    return peaks_x, peaks_y


def draw_graph(x0, y0, x1, y1, x2, y2):
    """Show datas as a graph."""

    fig = plt.figure()
    # Scale dierctions.
    plt.tick_params(which='both', direction='in')
    # Draw grid lines.
    plt.grid(which='major', axis='both')
    # Plot datas.
    plt.plot(x0,
             y0,
             linestyle='-',
             label='original data',
             )
    plt.plot(x1,
             y1,
             linestyle='',
             marker='o',
             markersize=10,
             markerfacecolor='none',
             markeredgewidth=2,
             label='peak polyfit',
             )
    plt.plot(x2,
             y2,
             linestyle='',
             marker='o',
             markersize=10,
             markerfacecolor='none',
             markeredgewidth=2,
             label='peaks find_peaks',
             )
    # Show the legend.
    plt.legend(loc='lower right', ncol=1, fontsize=10)
    # Show the graph.
    plt.show(block=False)
    print('\n''Press [enter] to close the graph.')
    input('-> ')
    plt.close()


if __name__ == '__main__':
    main()
