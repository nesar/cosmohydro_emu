"""
Static metadata for every emulated statistic.

This is the single registry the rest of the package reads from: which
statistics exist, how many input parameters each one takes, how the raw
emulator output maps to physical units, and how to label plots.

Redshift coverage and x-grids are *not* hard-coded here -- they are read from
the shipped ``data/<STAT>_training_data.npz`` files, which are the ground
truth for what was actually trained.
"""

# Physical parameter scaling factors (design values are divided by these)
SEED_MASS_SCALE = 1e6
VKIN_SCALE = 1e4
EPS_SCALE = 1e1

# Project fiducial cosmology (omega_m = Omega_m h^2)
FIDUCIAL_COSMOLOGY = {'omega_m': 0.14176, 'sigma_8': 0.8102}

# ---------------------------------------------------------------------------
# Input parameters
# ---------------------------------------------------------------------------
PARAMETERS = [
    # key,          latex,                              range,           description
    ('kappa_w',     r'$\kappa_\mathrm{w}$',             (2.0, 4.0),      'AGN wind coupling / wind efficiency parameter'),
    ('e_w',         r'$e_\mathrm{w}$',                  (0.2, 1.0),      'AGN energy efficiency / wind energy fraction'),
    ('M_seed',      r'$M_\mathrm{seed}/10^{6}$',        (0.6, 2.0),      'Black hole seed mass (in 10^6 M_sun)'),
    ('v_kin',       r'$v_\mathrm{kin}/10^{4}$',         (0.1, 1.2),      'Kinetic feedback velocity (in 10^4 km/s)'),
    ('epsilon_kin', r'$\epsilon_\mathrm{kin}/10^{1}$',  (0.02, 1.2),     'Kinetic feedback efficiency (in 10^1)'),
    ('omega_m',     r'$\omega_\mathrm{m}$',             (0.12, 0.155),   'Physical matter density omega_m = Omega_m h^2'),
    ('sigma_8',     r'$\sigma_8$',                      (0.7, 0.9),      'Amplitude of matter fluctuations'),
]

PARAM_SCALES = {'M_seed': SEED_MASS_SCALE, 'v_kin': VKIN_SCALE, 'epsilon_kin': EPS_SCALE}

# Which parameter columns each parameter-set uses
PARAM_SETS = {
    'subgrid+cosmo': [0, 1, 2, 3, 4, 5, 6],
    'cosmo': [5, 6],
}

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
# output_transform describes how the *raw* emulator output relates to the
# physical quantity returned by ``predict``:
#   None        : raw output is the physical quantity
#   'log10'     : emulator was trained on 10**y_phys  -> return log10(raw)
#   'pow10'     : emulator was trained on log10(y_phys) -> return 10**raw
# Uncertainties are propagated through the transform with the delta method.

STATISTICS = {
    # --- galaxy / halo / matter summary statistics ---------------------------
    'GSMF': dict(
        category='summary',
        param_set='subgrid+cosmo',
        output_transform='log10',
        title='Galaxy stellar mass function',
        xlabel=r'$M_{\ast}$ [$\mathrm{M}_{\odot}$]',
        ylabel=r'$\mathrm{d}n \, / \, \mathrm{d}\log_{10} M_{\ast} \; [(h^{-1}\mathrm{Mpc})^{-3}]$',
        xscale='log', yscale='log',
        x_description=r'Stellar mass [$M_\odot$]',
        valid_range=(5e9, 3e11),
    ),
    'HMF': dict(
        category='summary',
        param_set='subgrid+cosmo',
        output_transform='log10',
        title='Halo mass function',
        xlabel=r'$M_{\mathrm{halo}}$ [$h^{-1}\mathrm{M}_{\odot}$]',
        ylabel=r'$\mathrm{d}n / \mathrm{d}\log_{10} M \; [(h^{-1}\mathrm{Mpc})^{-3}]$',
        xscale='log', yscale='log',
        x_description=r'Halo mass $M_{\rm SO}$ [$h^{-1} M_\odot$]',
        valid_range=(2e11, 1e15),
    ),
    'fGas': dict(
        category='summary',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster gas fraction',
        xlabel=r'$M_{\mathrm{500c}}$ [$h^{-1}\mathrm{M}_{\odot}$]',
        ylabel=r'$M_{\mathrm{gas}} / M_{\mathrm{500c}} \quad [<R_{\mathrm{500c}}]$',
        xscale='log', yscale='linear',
        x_description=r'Halo mass $M_{500c}$ [$h^{-1} M_\odot$]',
        valid_range=(10**13.5, 10**14.3),
    ),
    'Pk-ratio': dict(
        category='summary',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Matter power spectrum suppression',
        xlabel=r'$k \, [h\,\mathrm{Mpc}^{-1}]$',
        ylabel=r'$P_{\mathrm{hydro}}(k)\,/\,P_{\mathrm{grav}}(k)$',
        xscale='log', yscale='linear',
        x_description=r'Wavenumber $k$ [$h$/Mpc]',
        valid_range=(0.015707963267948967, 8.042477193189871),
    ),
    'CSFR': dict(
        category='summary',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cosmic star formation rate',
        xlabel=r'$a$',
        ylabel=r'$\mathrm{CSFR} \, [\mathrm{M}_{\odot} \, \mathrm{yr}^{-1} \, (h^{-1}\mathrm{Mpc})^{-3}]$',
        xscale='linear', yscale='linear',
        x_description=r'Scale factor $a$',
        valid_range=(0.08, 1.0),
    ),
    # --- cluster radial profiles (stacked, R500c-scaled) ----------------------
    'CGD': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster gas density profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$\rho_{\mathrm{gas}} \,/\, \rho_{\mathrm{crit}}$',
        xscale='log', yscale='log',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    'CGED': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster gas electron density profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$n_{\mathrm{e}}$ [$\mathrm{cm}^{-3}$]',
        xscale='log', yscale='log',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    'CPP': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster gas pressure profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$P / P_{\mathrm{500}}$',
        xscale='log', yscale='log',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    'CTP': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster gas temperature profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$T / T_{\mathrm{500}}$',
        xscale='log', yscale='linear',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    'CEP': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster gas entropy profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$K / K_{\mathrm{500}}$',
        xscale='log', yscale='log',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    'CEEP': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster electron entropy profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$K_{\mathrm{e}} / K_{\mathrm{500}}$',
        xscale='log', yscale='log',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    'CMP': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster gas metallicity profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$Z / Z_{\odot}$',
        xscale='log', yscale='linear',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    'CYP': dict(
        category='profile',
        param_set='subgrid+cosmo',
        output_transform=None,
        title='Cluster Compton-y profile',
        xlabel=r'$r/R_{\mathrm{500c}}$',
        ylabel=r'$y_{\mathrm{SZ}}$',
        xscale='log', yscale='log',
        x_description=r'Radius $r/R_{500c}$',
        valid_range=(0.015, 2.75),
    ),
    # --- gravity-only (cosmology-only inputs) ---------------------------------
    'Pk_GO': dict(
        category='gravity_only',
        param_set='cosmo',
        output_transform='pow10',
        title='Gravity-only matter power spectrum',
        xlabel=r'$k \, [h\,\mathrm{Mpc}^{-1}]$',
        ylabel=r'$P_{\mathrm{grav}}(k) \, [(h^{-1}\mathrm{Mpc})^{3}]$',
        xscale='log', yscale='log',
        x_description=r'Wavenumber $k$ [$h$/Mpc]',
        valid_range=(0.015707963267948967, 8.042477193189871),
    ),
}

CATEGORY_LABELS = {
    'summary': 'summary statistics (5 subgrid + 2 cosmology parameters)',
    'profile': 'cluster profiles (5 subgrid + 2 cosmology parameters)',
    'gravity_only': 'gravity-only statistics (2 cosmology parameters)',
}


def get_stat_metadata(stat_name):
    """Return the metadata dict for ``stat_name`` (raises ``ValueError`` if unknown)."""
    if stat_name not in STATISTICS:
        raise ValueError(
            f"Unknown statistic: {stat_name!r}\n"
            f"Available: {list(STATISTICS)}"
        )
    return STATISTICS[stat_name]


def get_param_columns(stat_name):
    """Design-matrix columns (into ``PARAMETERS``) used by ``stat_name``."""
    return PARAM_SETS[get_stat_metadata(stat_name)['param_set']]
