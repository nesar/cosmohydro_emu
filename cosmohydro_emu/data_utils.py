"""
Data utilities for the CosmoHydro emulator package.

Information about the independent variables (x-grids), the trained
redshifts, plotting labels and the input parameters of each statistic.
"""

import os

import numpy as np

from .emulator import get_data_path
from .model_metadata import (
    FIDUCIAL_COSMOLOGY,
    PARAM_SCALES,
    PARAMETERS,
    STATISTICS,
    get_param_columns,
    get_stat_metadata,
)


def _load(stat_name):
    d = np.load(get_data_path(stat_name), allow_pickle=False)
    return {k: d[k] for k in d.files}


def get_test_data(stat_name, physical=True):
    """
    Load the 10 held-out simulations (runs 100-109) of a summary statistic.

    They are built by the same preprocessing as the training set
    (``get_data_path(stat_name)``), on the same x-grid and snapshots, and were
    never used to train the emulator.

    Parameters
    ----------
    stat_name : str
    physical : bool, optional
        If True (default) apply the emulator's output transform, so ``y`` is
        directly comparable to ``emulator.predict``.  If False, return the raw
        training-space values stored in the file.

    Returns
    -------
    dict
        ``'params'`` (10, n_params) scaled inputs, ``'y'`` (10, n_z, n_x),
        ``'x_grid'`` (n_x,), ``'redshifts'`` (n_z,) ascending.

    Examples
    --------
    >>> t = get_test_data('GSMF')
    >>> mean, std = load_emulator('GSMF').predict(t['params'], z=t['redshifts'][0])
    >>> # compare mean.T with t['y'][:, 0, :]
    """
    meta = get_stat_metadata(stat_name)
    path = os.path.join(os.path.dirname(get_data_path(stat_name)),
                        f'{stat_name}_test_data.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data not found: {path}")
    with np.load(path, allow_pickle=False) as d:
        d = {k: d[k] for k in d.files}
    order = np.argsort(d['redshifts'])
    y = d['y_vals'][:, order, :]
    if physical:
        t = meta['output_transform']
        if t == 'log10':
            y = np.log10(y)
        elif t == 'pow10':
            y = 10.0 ** y
    return {'params': d['p_test'], 'y': y, 'x_grid': d['y_ind'],
            'redshifts': d['redshifts'][order]}


def get_x_grid(stat_name):
    """
    Get the independent-variable grid for a summary statistic.

    This is the grid the emulator was trained on and on which ``predict``
    returns values.

    Parameters
    ----------
    stat_name : str

    Returns
    -------
    x_grid : np.ndarray
    x_label : str
        Short description of the independent variable.

    Examples
    --------
    >>> x_grid, x_label = get_x_grid('GSMF')
    """
    meta = get_stat_metadata(stat_name)
    return _load(stat_name)['y_ind'], meta['x_description']


def get_redshifts(stat_name):
    """
    Get the trained snapshot redshifts of a summary statistic (ascending).

    Parameters
    ----------
    stat_name : str

    Returns
    -------
    np.ndarray
    """
    return np.sort(_load(stat_name)['redshifts'])


def get_plot_info(stat_name):
    """
    Get plotting information for a summary statistic.

    Returns
    -------
    dict
        Keys: ``'title'``, ``'xlabel'``, ``'ylabel'``, ``'xscale'``, ``'yscale'``.
    """
    meta = get_stat_metadata(stat_name)
    return {k: meta[k] for k in ('title', 'xlabel', 'ylabel', 'xscale', 'yscale')}


def get_valid_range(stat_name):
    """
    Get the recommended range of the independent variable for a statistic.

    Returns
    -------
    tuple
        ``(min_value, max_value)``
    """
    return tuple(get_stat_metadata(stat_name)['valid_range'])


def get_parameter_info(stat_name=None):
    """
    Get information about the input parameters.

    Parameters
    ----------
    stat_name : str, optional
        If given, restrict to the parameters used by that statistic
        (7 for most, 2 cosmology parameters for gravity-only statistics).

    Returns
    -------
    dict
        Keys: ``'names'``, ``'latex_names'``, ``'ranges'``, ``'descriptions'``,
        ``'scales'``, ``'fiducial'``.
    """
    cols = get_param_columns(stat_name) if stat_name is not None else range(len(PARAMETERS))
    params = [PARAMETERS[i] for i in cols]
    names = [p[0] for p in params]
    return {
        'names': names,
        'latex_names': [p[1] for p in params],
        'ranges': {p[0]: p[2] for p in params},
        'descriptions': {p[0]: p[3] for p in params},
        'scales': {k: v for k, v in PARAM_SCALES.items() if k in names},
        'fiducial': {k: v for k, v in FIDUCIAL_COSMOLOGY.items() if k in names},
    }


def get_statistic_info(stat_name=None):
    """
    Summary table of one or all statistics.

    Returns
    -------
    dict
        ``stat_name -> {'title', 'category', 'n_params', 'redshifts', 'n_x', 'x_description'}``
    """
    stats = [stat_name] if stat_name is not None else list(STATISTICS)
    out = {}
    for s in stats:
        meta = get_stat_metadata(s)
        d = _load(s)
        out[s] = {
            'title': meta['title'],
            'category': meta['category'],
            'n_params': len(get_param_columns(s)),
            'redshifts': np.sort(d['redshifts']),
            'n_x': int(d['y_ind'].size),
            'x_description': meta['x_description'],
        }
    return out
