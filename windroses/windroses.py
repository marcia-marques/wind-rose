import numpy as np
from math import radians
import matplotlib.pyplot as plt


def wind_rose(df, wd, nbins=16, wind=True, xticks=8, plot=111,
              ylim=False, yaxis=False, yticks=False):
    """
    Return a wind rose.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame holding the data.
    wd : str
        Wind direction column name.
    nbins : int, optional
        Number of bins to plot, default is 16.
    xticks : int {4, 8, 16} , optional
        Number of xticks, default is 8.
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
    plot : int, optional
        nrows, ncols, index to define subplots, default is 111,
        it is used to plot seasonal wind roses.
    ylim : int or float, optional
        Maximum limit for y-axis, default is False.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    yticks : list-like, optional
        List of yticks, default is False.
    """

    labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'SSE', 'SE', 'SSE',
              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # adjust wind direction (align North)
    x = 360 - (180 / nbins)
    w_dir = np.zeros(len(df[wd]))
    for i in range(len(df[wd])):
        if x <= df[wd][i] <= 360:
            w_dir[i] = df[wd][i] - 360
        else:
            w_dir[i] = df[wd][i]
    df['dir'] = w_dir

    # histogram
    bins = np.arange(- (180 / nbins), 360 + (180 / nbins), 360 / nbins)
    n, bins = np.histogram(df.dir, bins=bins)

    # wind rose
    ax = plt.subplot(plot, projection='polar')
    ax.bar([radians(x + (180 / nbins)) for x in bins][:-1],
           n,
           width=2 * np.pi / nbins)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    plt.grid(axis='both', which='major', linestyle='--')

    # categorical xticklabels
    if xticks == 4:
        ax.set_xticks([radians(x) for x in np.arange(0, 360, 360 / xticks)])
        if wind:
            ax.set_xticklabels([x for x in labels[::4]])
    elif xticks == 8:
        ax.set_xticks([radians(x) for x in np.arange(0, 360, 360 / xticks)])
        if wind:
            ax.set_xticklabels([x for x in labels[::2]])
    elif xticks == 16:
        ax.set_xticks([radians(x) for x in np.arange(0, 360, 360 / xticks)])
        if wind:
            ax.set_xticklabels(labels)
    else:
        raise Exception("xticks should be 4, 8, or 16")

    # y axis limit
    if ylim:
        plt.ylim(0, ylim)

    # y axis position
    if yaxis:
        ax.set_rlabel_position(yaxis)

    # y axis ticks
    if yticks:
        ax.set_yticks(yticks)

    return
