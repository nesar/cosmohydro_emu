"""
CosmoHydro Emulator Package

Gaussian-process emulators for summary statistics of the CRK-HACC CosmoHydro
simulation suite: 110 hydrodynamical simulations (L = 400 h^-1 Mpc) varying
5 subgrid-physics parameters and 2 cosmological parameters, with outputs at
multiple redshifts.

Examples
--------
>>> from cosmohydro_emu import load_emulator
>>>
>>> emu = load_emulator('GSMF')
>>> # [kappa_w, e_w, M_seed/1e6, v_kin/1e4, eps_kin/1e1, omega_m, sigma_8]
>>> params = [3.0, 0.5, 1.0, 0.65, 0.5, 0.14176, 0.8102]
>>> mean, std = emu.predict(params, z=0.5)
>>> emu.redshifts          # trained snapshots this emulator can interpolate between

Available statistics
--------------------
Summary statistics (7 parameters):
- GSMF : galaxy stellar mass function            (z = 0 -- 2)
- HMF  : halo mass function                      (z = 0 -- 2)
- fGas : cluster gas fraction                    (z = 0 -- 1)
- Pk-ratio : matter power spectrum suppression ratio (z = 0, 0.1, 0.5, 1, 2)
- CSFR : cosmic star formation history           (z = 0 output, history in a)

Cluster profiles (7 parameters, z = 0 -- 0.5):
- CGD, CGED, CPP, CTP, CEP, CEEP, CMP, CYP

Gravity-only (2 cosmology parameters):
- Pk_GO : gravity-only matter power spectrum     (z = 0, 0.1, 0.5, 1, 2)
"""

__version__ = '0.1.0'
__author__ = 'Nesar Ramachandra'

from .emulator import (
    CosmoHydroEmulator,
    load_emulator,
    list_available_statistics,
    PARAM_NAMES,
    PARAM_KEYS,
    AVAILABLE_STATS,
    AVAILABLE_STATS_SUMMARY,
    AVAILABLE_STATS_PROFILE,
    AVAILABLE_STATS_GRAVITY_ONLY,
)

from .model_metadata import (
    SEED_MASS_SCALE,
    VKIN_SCALE,
    EPS_SCALE,
    FIDUCIAL_COSMOLOGY,
)

from .data_utils import (
    get_x_grid,
    get_redshifts,
    get_plot_info,
    get_valid_range,
    get_parameter_info,
    get_statistic_info,
)

__all__ = [
    # Main classes and functions
    'CosmoHydroEmulator',
    'load_emulator',
    'list_available_statistics',

    # Constants
    'PARAM_NAMES',
    'PARAM_KEYS',
    'AVAILABLE_STATS',
    'AVAILABLE_STATS_SUMMARY',
    'AVAILABLE_STATS_PROFILE',
    'AVAILABLE_STATS_GRAVITY_ONLY',
    'SEED_MASS_SCALE',
    'VKIN_SCALE',
    'EPS_SCALE',
    'FIDUCIAL_COSMOLOGY',

    # Data utilities
    'get_x_grid',
    'get_redshifts',
    'get_plot_info',
    'get_valid_range',
    'get_parameter_info',
    'get_statistic_info',
]
