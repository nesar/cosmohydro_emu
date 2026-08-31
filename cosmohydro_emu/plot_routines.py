"""
Plotting helpers (optional; requires matplotlib).
"""

import numpy as np

from .data_utils import get_plot_info, get_x_grid


def plot_prediction(emu, params, z=0.0, ax=None, n_sigma=2.0, label=None, **plot_kw):
    """
    Plot an emulator prediction with its uncertainty band.

    Parameters
    ----------
    emu : CosmoHydroEmulator
    params : array-like
        One parameter vector.
    z : float
        Redshift.
    ax : matplotlib.axes.Axes, optional
    n_sigma : float
        Half-width of the shaded band in units of the GP standard deviation.
    label : str, optional
    **plot_kw
        Passed to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    mean, std = emu.predict(params, z=z)
    x_grid, _ = get_x_grid(emu.stat_name)
    info = get_plot_info(emu.stat_name)

    line, = ax.plot(x_grid, mean, lw=2, label=label if label is not None else f'z = {z:.2f}', **plot_kw)
    ax.fill_between(x_grid, mean - n_sigma * std, mean + n_sigma * std,
                    alpha=0.25, color=line.get_color(), lw=0)
    ax.set_xscale(info['xscale'])
    ax.set_yscale(info['yscale'])
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])
    return ax


def plot_redshift_evolution(emu, params, redshifts=None, ax=None, cmap='viridis'):
    """
    Plot the prediction for one parameter set at several redshifts.

    Parameters
    ----------
    emu : CosmoHydroEmulator
    params : array-like
    redshifts : array-like, optional
        Default: all trained snapshot redshifts.
    ax : matplotlib.axes.Axes, optional
    cmap : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    redshifts = emu.redshifts if redshifts is None else np.atleast_1d(redshifts)
    x_grid, _ = get_x_grid(emu.stat_name)
    info = get_plot_info(emu.stat_name)
    norm = mcolors.Normalize(vmin=redshifts.min(), vmax=redshifts.max())
    colors = cm.get_cmap(cmap)(norm(redshifts))
    for z, c in zip(redshifts, colors):
        mean, _ = emu.predict(params, z=z)
        ax.plot(x_grid, mean, lw=1.8, color=c, label=f'z = {z:.2f}')
    ax.set_xscale(info['xscale'])
    ax.set_yscale(info['yscale'])
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])
    ax.legend(fontsize='small', ncol=2)
    return ax


def plot_scatter_matrix(df, colors):
    """Scatter-matrix of a design (pandas DataFrame) -- kept from ``subgrid_emu``."""
    import matplotlib.pyplot as plt
    import pandas as pd

    f, a = plt.subplots(1, 1, figsize=(10, 10))
    scatter_matrix = pd.plotting.scatter_matrix(df, color=colors, figsize=(10, 10), alpha=1.0,
                                                ax=a, grid=False, diagonal='hist',
                                                range_padding=0.1, s=80)
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=14, rotation=0)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14, rotation=90)
    return f
