import numpy as np
from math import radians
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def wind_rose(df, wd, nbins=16, xticks=8, plot=111, wind=True, ylim=False, yaxis=False, yticks=False):
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
    plot : int, optional
        nrows, ncols, index to define subplots, default is 111,
        it is used to plot seasonal wind roses.
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
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


def wind_rose_season(df, wd, nbins=16, xticks=8, wind=True, south=True, ylim=False, yaxis=False, yticks=False):
    """
    Return a wind rose for each season.

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
    south : bool, optional, default is True
        If True, seasons are calculated to Southern Hemisphere, otherwise Northern Hemisphere.
    ylim : int or float, optional
        Maximum limit for y-axis, default is False.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    yticks : list-like, optional
        List of yticks, default is False.
    """

    # create a new column season
    if south:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Summer', 2: 'Autumn', 3: 'Winter', 4: 'Spring'})
    else:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'})

    # windroses
    for i, season in enumerate(df['season'].unique()):
        df_season = df.copy()
        df_season = df_season.loc[df_season['season'] == season]
        wind_rose(df_season, wd, nbins=nbins, xticks=xticks, wind=wind, plot=221+i,
                  ylim=ylim, yaxis=yaxis, yticks=yticks)
        plt.title(season + '\n', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return


def wind_rose_scatter(df, ws, wd, xticks=8, size=2, plot=111, wind=True, ylim=False, yaxis=False, yticks=False):
    """
    Return a scatter polar plot of wind speed.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame holding the data.
    ws : str
        Wind speed column name.
    wd : str
        Wind direction column name.
    xticks : int {4, 8, 16} , optional
        Number of xticks, default is 8.
    size : int or float, default os 2
        Size of scatter points.
    plot : int, optional
        nrows, ncols, index to define subplots, default is 111,
        it is used to plot seasonal wind roses.
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
    ylim : int or float, optional
        Maximum limit for y-axis, default is False.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    yticks : list-like, optional
        List of yticks, default is False.
    """

    labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'SSE', 'SE', 'SSE',
              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # windrose
    ax = plt.subplot(plot, projection='polar')
    ax.scatter(x=[radians(x) for x in df[wd].values], y=df[ws].values, s=size)
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


def wind_rose_scatter_season(df, ws, wd, xticks=8, size=2, wind=True, south=True,
                             ylim=False, yaxis=False, yticks=False):
    """
    Return a scatter polar plot of wind speed.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame holding the data.
    ws : str
        Wind speed column name.
    wd : str
        Wind direction column name.
    xticks : int {4, 8, 16} , optional
        Number of xticks, default is 8.
    size : int or float, default os 2
        Size of scatter points.
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
    south : bool, optional, default is True
        If True, seasons are calculated to Southern Hemisphere, otherwise Northern Hemisphere.
    ylim : int or float, optional
        Maximum limit for y-axis, default is False.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    yticks : list-like, optional
        List of yticks, default is False.
    """

    # create a new column season
    if south:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Summer', 2: 'Autumn', 3: 'Winter', 4: 'Spring'})
    else:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'})

    # windroses
    for i, season in enumerate(df['season'].unique()):
        df_season = df.copy()
        df_season = df_season.loc[df_season['season'] == season]
        wind_rose_scatter(df_season, ws=ws, wd=wd, xticks=xticks, size=size, plot=221+i, wind=wind,
                          ylim=ylim, yaxis=yaxis, yticks=yticks)
        plt.title(season + '\n', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return


def wind_rose_speed(df, ws, wd, nbins=16, xticks=8, plot=111, wind=True, ylim=False, yaxis=False, yticks=False,
                    lims=False, loc='lower left'):
    """
    Return a wind rose with wind speed ranges.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame holding the data.
    ws : str
        Wind speed column name.
    wd : str
        Wind direction column name.
    nbins : int, optional
        Number of bins to plot, default is 16.
    xticks : int {4, 8, 16} , optional
        Number of xticks, default is 8.
    plot : int, optional
        nrows, ncols, index to define subplots, default is 111,
        it is used to plot seasonal wind roses.
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
    ylim : int or float, optional
        Maximum limit for y-axis, default is False.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    yticks : list-like, optional
        List of yticks, default is False.
    lims : list-like, optional, default is False.
        Wind speed ranges.
    loc : int or str, optional, default is 'lower left'
        Legend location.
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

    # bins
    bins = np.arange(- (180 / nbins), 360 + (180 / nbins), 360 / nbins)

    # default wind speed limits
    if not lims:
        lims = np.linspace(df[ws].min(), df[ws].max(), num=5, endpoint=False)
        lims = np.append(lims, df.ws.max())

    # matrix to store n values for all ranges
    ns = np.zeros((len(bins) - 1, len(lims) - 1))

    # histogram
    for i in range(len(lims) - 1):
        ds = df.copy()
        if i == len(lims) - 2:
            ds = ds[(df[ws] >= lims[i]) & (ds[ws] <= lims[i + 1])]
        else:
            ds = ds[(df[ws] >= lims[i]) & (ds[ws] < lims[i + 1])]
        n, bins = np.histogram(ds.dir, bins=bins)
        ns[:, i] = n

    if np.sum(ns) != df.dir.count():
        print("Warning: wind speed range does not cover all data")

    # windrose
    ax = plt.subplot(plot, projection='polar')
    for i in range(len(lims) - 1):
        ax.bar([radians(x + (180 / nbins)) for x in bins][:-1],
               np.sum(ns[:, 0:len(lims) - 1 - i], axis=1),
               width=2 * np.pi / nbins,
               label="{:.1f}".format(lims[len(lims) - 1 - i - 1]) + ' - ' +
                     "{:.1f}".format(lims[len(lims) - 1 - i]))
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

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc=loc)

    return


def wind_rose_speed_season(df, ws, wd, nbins=16, xticks=8, wind=True, south=True, ylim=False, yaxis=False, yticks=False,
                           lims=False, loc='lower left'):
    """
    Return a wind rose with wind speed ranges for each season.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame holding the data.
    ws : str
        Wind speed column name.
    wd : str
        Wind direction column name.
    nbins : int, optional
        Number of bins to plot, default is 16.
    xticks : int {4, 8, 16} , optional
        Number of xticks, default is 8.
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
    south : bool, optional, default is True
        If True, seasons are calculated to Southern Hemisphere, otherwise Northern Hemisphere.
    ylim : int or float, optional
        Maximum limit for y-axis, default is False.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    yticks : list-like, optional
        List of yticks, default is False.
    lims : list-like, optional, default is False.
        Wind speed ranges.
    loc : int or str, optional, default is 'lower left'
        Legend location.
    """

    # create a new column season
    if south:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Summer', 2: 'Autumn', 3: 'Winter', 4: 'Spring'})
    else:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'})

    # windroses
    for i, season in enumerate(df['season'].unique()):
        df_season = df.copy()
        df_season = df_season.loc[df_season['season'] == season]
        wind_rose_speed(df_season, ws, wd, nbins=nbins, xticks=xticks, wind=wind, plot=221+i,
                        ylim=ylim, yaxis=yaxis, yticks=yticks, lims=lims, loc=loc)
        plt.title(season + '\n', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return


def wind_rose_pollution(df, var, ws, wd, var_label, cmap='viridis', nbins=16, min_bin=1, contrib=False,
                        xticks=8, plot=111, z_values=None, wind=True, yaxis=False, lims=False):
    """
    Return a wind rose for pollutant concentration.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame holding the data.
    var : str
        Pollutant column name.
    ws : str
        Wind speed column name.
    wd : str
        Wind direction column name.
    var_label : str
        Pollutant label.
    cmap : str or Colormap
        A Colormap instance or registered colormap name, default is 'viridis'.
    nbins : int, optional
        Number of bins to plot, default is 16.
    min_bin : int, optional
        The minimum number of points allowed in a wind speed/wind
        direction bin, default is 1.
    contrib : bool, optional
        If true, return the percentage of contribution of each segment,
        concentration of the pollutant weighted by wind speed/direction,
        default is False.
    xticks : int {4, 8, 16} , optional
        Number of xticks, default is 8.
    plot : int, optional
        nrows, ncols, index to define subplots, default is 111,
        it is used to plot seasonal wind roses.
    z_values : list-like, optional, default is None
        Min and max values for z values (colorbar).
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    lims : list-like, optional, default is False.
        Wind speed ranges.
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

    # bins
    bins = np.arange(- (180 / nbins), 360 + (180 / nbins), 360 / nbins)

    # default wind speed limits
    if not lims:
        lims = np.linspace(df[ws].min(), df[ws].max(), num=5, endpoint=False)
        lims = np.append(lims, df.ws.max())

    # matrix to store concentration values for all ranges
    ns = np.zeros((len(lims) - 1, len(bins) - 1))

    # histogram
    # wind speed ranges
    for i in range(len(lims) - 1):
        ds = df.copy()
        if i == len(lims) - 2:
            ds = ds[(ds[ws] >= lims[i]) & (ds[ws] <= lims[i + 1])]
        else:
            ds = ds[(ds[ws] >= lims[i]) & (ds[ws] < lims[i + 1])]

        # wind direction bins
        for j in range(len(bins) - 1):
            ds = ds[(ds['dir'] >= bins[j]) & (ds['dir'] < bins[j + 1])]
            if ds[var].count() >= min_bin:
                ns[i, j] = ds[var].mean()
                if contrib and min_bin == 1:
                    weight = ds[var].mean() / df[(df[ws] > 0) | (~np.isnan(df[wd]))][var].mean()
                    ns[i, j] = 100 * weight * ds[var].count() / df[(df[ws] > 0) | (~np.isnan(df[wd]))][var].count()
                else:
                    raise Exception("to use contrib option, min_bin must be 1")
            else:
                ns[i, j] = np.nan
            ds = df.copy()
            if i == len(lims) - 2:
                ds = ds[(ds[ws] >= lims[i]) & (ds[ws] <= lims[i + 1])]
            else:
                ds = ds[(ds[ws] >= lims[i]) & (ds[ws] < lims[i + 1])]

    # windrose
    ax = plt.subplot(plot, projection='polar')
    if z_values:
        cf = ax.pcolormesh(np.radians(bins),
                           lims, ns,
                           shading='flat', zorder=0,
                           vmin=z_values[0],
                           vmax=z_values[1])
    else:
        cf = ax.pcolormesh(np.radians(bins),
                           lims, ns,
                           shading='flat', zorder=0,
                           cmap=cmap)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    cbar = plt.colorbar(cf, ax=ax, pad=0.1, shrink=0.75, format='%.0f%%')
    cbar.set_label(var_label)
    ax.set_yticks(lims)
    bbox = dict(boxstyle="round", ec=None, fc="white", alpha=0.5)
    plt.setp(ax.get_yticklabels(), bbox=bbox)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_xticks(np.radians(np.arange((180 / nbins), 360 + (180 / nbins), 360 / nbins)), minor=True)
    plt.grid(axis='x', which='minor', linestyle='-', linewidth=0.25)
    plt.grid(axis='y', which='major', linestyle='-', linewidth=0.55)

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

    # y axis position
    if yaxis:
        ax.set_rlabel_position(yaxis)

    return


def wind_rose_pollution_season(df, var, ws, wd, var_label, cmap='viridis', nbins=16, min_bin=1, contrib=False,
                               xticks=8, z_values=None, wind=True, south=True, yaxis=False, lims=False):
    """
    Return a wind rose for pollutant concentration for each season.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame holding the data.
    var : str
        Pollutant column name.
    ws : str
        Wind speed column name.
    wd : str
        Wind direction column name.
    var_label : str
        Pollutant label.
    cmap : str or Colormap
        A Colormap instance or registered colormap name, default is 'viridis'.
    nbins : int, optional
        Number of bins to plot, default is 16.
    min_bin : int, optional
        The minimum number of points allowed in a wind speed/wind
        direction bin, default is 1.
    contrib : bool, optional
        If true, return the percentage of contribution of each segment,
        concentration of the pollutant weighted by wind speed/direction,
        default is False.
    xticks : int {4, 8, 16} , optional
        Number of xticks, default is 8.
    z_values : list-like, optional, default is None
        Min and max values for z values (colorbar).
    wind : bool, optional
        Show cardinal directions (i.e. ['N', 'NE', ...]), defaults is True.
    south : bool, optional, default is True
        If True, seasons are calculated to Southern Hemisphere, otherwise Northern Hemisphere.
    yaxis : int or flot, optional
        Position of y-axis (0 - 360), default is False.
    lims : list-like, optional, default is False.
        Wind speed ranges.
    """

    # create a new column season
    if south:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Summer', 2: 'Autumn', 3: 'Winter', 4: 'Spring'})
    else:
        df['season'] = ((df.index.month % 12 + 3) // 3).map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'})

    # windroses
    for i, season in enumerate(df['season'].unique()):
        df_season = df.copy()
        df_season = df_season.loc[df_season['season'] == season]
        wind_rose_pollution(df_season, var, ws, wd, var_label, cmap=cmap, nbins=nbins, min_bin=min_bin, contrib=contrib,
                            xticks=xticks, plot=221+i, z_values=z_values, wind=wind, yaxis=yaxis, lims=lims)
        plt.title(season + '\n', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return
