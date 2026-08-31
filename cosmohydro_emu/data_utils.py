"""
Data utilities for the CosmoHydro emulator package.

Information about the independent variables (x-grids), the trained
redshifts, plotting labels and the input parameters of each statistic.
"""

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
