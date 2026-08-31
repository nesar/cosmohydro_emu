"""
CosmoHydro Emulator

Loads the trained multi-redshift Gaussian-process emulators (SEPIA) and
exposes a single ``predict(params, z=...)`` interface for every statistic.
"""

import contextlib
import io
import os
import pickle
import warnings

import numpy as np
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPredict import SepiaEmulatorPrediction

from .model_metadata import (
    PARAMETERS,
    STATISTICS,
    get_param_columns,
    get_stat_metadata,
)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_PKG_DIR, 'data')
_MODEL_DIR = os.path.join(_PKG_DIR, 'models')

# Redshifts closer than this to a trained snapshot use that snapshot directly
_Z_TOL = 1e-4

# Parameter names (LaTeX) for the full 7-parameter design, in design order
PARAM_NAMES = [p[1] for p in PARAMETERS]
PARAM_KEYS = [p[0] for p in PARAMETERS]

# Available summary statistics, grouped
AVAILABLE_STATS_SUMMARY = [s for s, m in STATISTICS.items() if m['category'] == 'summary']
AVAILABLE_STATS_PROFILE = [s for s, m in STATISTICS.items() if m['category'] == 'profile']
AVAILABLE_STATS_GRAVITY_ONLY = [s for s, m in STATISTICS.items() if m['category'] == 'gravity_only']
AVAILABLE_STATS = AVAILABLE_STATS_SUMMARY + AVAILABLE_STATS_PROFILE + AVAILABLE_STATS_GRAVITY_ONLY


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def get_data_path(stat_name):
    """Path to the shipped training-data ``.npz`` for ``stat_name``."""
    get_stat_metadata(stat_name)
    path = os.path.join(_DATA_DIR, f'{stat_name}_training_data.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found: {path}")
    return path


def get_model_path(stat_name, z_index):
    """Path (without ``.pkl``) to the trained SEPIA model for one snapshot."""
    get_stat_metadata(stat_name)
    base = os.path.join(_MODEL_DIR, stat_name, f'multivariate_model_z_index{z_index}')
    if not os.path.exists(base + '.pkl'):
        raise FileNotFoundError(f"Model file not found: {base}.pkl")
    return base


def _load_training_data(stat_name):
    d = np.load(get_data_path(stat_name), allow_pickle=False)
    return {k: d[k] for k in d.files}


def _n_pc_from_pickle(model_base):
    """Number of PCA basis vectors the saved model was trained with.

    Read from ``samples['betaU'].shape[2]`` so the reconstructed K basis always
    matches what the stored MCMC samples expect.  Returns ``None`` if the file
    doesn't expose it.
    """
    try:
        with open(model_base + '.pkl', 'rb') as fh:
            blob = pickle.load(fh)
    except Exception:
        return None
    samples = blob.get('samples') if isinstance(blob, dict) else None
    betaU = samples.get('betaU') if isinstance(samples, dict) else None
    if betaU is None or getattr(betaU, 'ndim', 0) < 3:
        return None
    return int(betaU.shape[2])


def _build_sepia_model(design, y_vals, y_ind, n_pc):
    """Recreate the SEPIA data container and PCA basis used at training time."""
    data = SepiaData(t_sim=design, y_sim=y_vals, y_ind_sim=y_ind)
    data.transform_xt()
    data.standardize_y()
    data.create_K_basis(n_pc=n_pc)
    return SepiaModel(data)


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------
class _SnapshotEmulator:
    """One trained SEPIA model at a single snapshot (internal)."""

    def __init__(self, model_base, design, y_vals, y_ind, exp_variance=None):
        n_pc = exp_variance if exp_variance is not None else _n_pc_from_pickle(model_base)
        if n_pc is None:
            n_pc = 0.99
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = _build_sepia_model(design, y_vals, y_ind, n_pc)
            self.model.restore_model_info(model_base)
        sim = self.model.data.sim_data
        self._K = sim.K              # (pu, p)
        self._K_T = sim.K.T
        # column vectors (p, 1) or (1, 1) so they broadcast against (p, n)
        self._y_sd = np.atleast_1d(np.asarray(sim.orig_y_sd, dtype=float)).reshape(-1, 1)
        self._y_mean = np.atleast_1d(np.asarray(sim.orig_y_mean, dtype=float)).reshape(-1, 1)
        self._samples = self.model.get_samples(numsamples=1)
        self.n_pc = int(self._K.shape[0])

    # SEPIA builds an (n_pred*pu)^2 covariance per call; keep batches modest.
    _CHUNK = 64

    def predict(self, params):
        """Analytic GP mean/std on the training y-scale. Shapes ``(p, n)``."""
        n = params.shape[0]
        if n > self._CHUNK:
            parts = [self._predict_chunk(params[i:i + self._CHUNK]) for i in range(0, n, self._CHUNK)]
            return (np.concatenate([p[0] for p in parts], axis=1),
                    np.concatenate([p[1] for p in parts], axis=1))
        return self._predict_chunk(params)

    def _predict_chunk(self, params):
        n = params.shape[0]
        pu = self.n_pc
        pred = SepiaEmulatorPrediction(t_pred=params, samples=self._samples,
                                       model=self.model, storeMuSigma=True)
        # SEPIA stores the latent GP mean PC-major: mu[j*n:(j+1)*n] is PC j at
        # every input; sigma is block-diagonal across PCs with (n x n) blocks.
        mu = pred.mu[0].reshape(pu, n)                                  # (pu, n)
        sig4 = pred.sigma[0].reshape(pu, n, pu, n)
        sig = np.einsum('piqi->pqi', sig4)                              # (pu, pu, n)

        y_mu = self._K_T @ mu                                           # (p, n)
        y_var = np.einsum('ap,pqi,qa->ai', self._K_T, sig, self._K)     # (p, n)
        y_std = np.sqrt(np.clip(y_var, 0, None))

        y_mu = self._y_sd * y_mu + self._y_mean
        y_std = self._y_sd * y_std
        return y_mu, y_std


class CosmoHydroEmulator:
    """
    Emulator for one summary statistic across all its trained redshifts.

    Parameters
    ----------
    stat_name : str
        Name of the summary statistic (see ``AVAILABLE_STATS``).
    redshifts : array-like, optional
        Subset of trained snapshot redshifts to load (default: all).  Values
        are matched to the nearest trained snapshot.
    exp_variance : float or int, optional
        Override the PCA basis size (explained-variance fraction or integer
        number of components).  By default it is read from each model pickle,
        which is what guarantees consistency with the stored GP samples.

    Attributes
    ----------
    stat_name : str
    n_params : int
        Number of input parameters (7, or 2 for gravity-only statistics).
    param_names : list of str
        LaTeX names of the expected input parameters, in order.
    redshifts : np.ndarray
        Trained snapshot redshifts (ascending).
    z_range : tuple
        ``(z_min, z_max)`` over which ``predict`` can interpolate.
    x_grid : np.ndarray
        Independent-variable grid the predictions are defined on.
    """

    def __init__(self, stat_name, redshifts=None, exp_variance=None):
        self.stat_name = stat_name
        self.meta = get_stat_metadata(stat_name)
        self._param_cols = get_param_columns(stat_name)
        self.n_params = len(self._param_cols)
        self.param_names = [PARAMETERS[i][1] for i in self._param_cols]
        self.param_keys = [PARAMETERS[i][0] for i in self._param_cols]
        self._param_lo = np.array([PARAMETERS[i][2][0] for i in self._param_cols])
        self._param_hi = np.array([PARAMETERS[i][2][1] for i in self._param_cols])
        self.exp_variance = exp_variance

        d = _load_training_data(stat_name)
        self.x_grid = d['y_ind']
        self.design = d['p_train']

        # sort trained snapshots by ascending redshift
        order = np.argsort(d['redshifts'])
        all_z = d['redshifts'][order]
        all_idx = d['z_index_range'][order]
        all_y = d['y_vals'][:, order, :]

        if redshifts is not None:
            sel = sorted({int(np.argmin(np.abs(all_z - z))) for z in np.atleast_1d(redshifts)})
        else:
            sel = list(range(len(all_z)))

        self.redshifts = all_z[sel]
        self.z_indices = all_idx[sel]
        self._snapshots = [
            _SnapshotEmulator(get_model_path(stat_name, int(all_idx[k])),
                              self.design, all_y[:, k, :], self.x_grid,
                              exp_variance=exp_variance)
            for k in sel
        ]
        self.z_range = (float(self.redshifts.min()), float(self.redshifts.max()))

    # ------------------------------------------------------------------
    @property
    def n_redshifts(self):
        return len(self.redshifts)

    @property
    def n_pc(self):
        """PCA basis size of each loaded snapshot model (ascending z)."""
        return [s.n_pc for s in self._snapshots]

    # ------------------------------------------------------------------
    def _check_params(self, params, check_bounds):
        params = np.atleast_2d(np.asarray(params, dtype=float))
        if params.ndim != 2 or params.shape[1] != self.n_params:
            raise ValueError(
                f"{self.stat_name}: expected {self.n_params} parameters "
                f"{self.param_keys}, got array of shape {params.shape}"
            )
        if check_bounds:
            below = params < self._param_lo
            above = params > self._param_hi
            if below.any() or above.any():
                bad = np.where(below.any(0) | above.any(0))[0]
                warnings.warn(
                    f"{self.stat_name}: parameter(s) "
                    f"{[self.param_keys[i] for i in bad]} outside the training "
                    f"range; predictions are extrapolations.",
                    RuntimeWarning, stacklevel=3,
                )
        return params

    def _predict_raw_at_z(self, params, z):
        """Raw (training-scale) mean/std at scalar redshift ``z``, shapes ``(p, n)``."""
        z = float(z)
        zmin, zmax = self.z_range
        if z < zmin - _Z_TOL or z > zmax + _Z_TOL:
            raise ValueError(
                f"{self.stat_name}: z={z} outside the trained range "
                f"[{zmin:.3f}, {zmax:.3f}] (snapshots at z = "
                f"{np.round(self.redshifts, 3).tolist()})"
            )
        k_near = int(np.argmin(np.abs(self.redshifts - z)))
        if abs(self.redshifts[k_near] - z) < _Z_TOL or self.n_redshifts == 1:
            return self._snapshots[k_near].predict(params)

        # linear interpolation between the two bracketing snapshots
        k_hi = int(np.searchsorted(self.redshifts, z))      # first snapshot with z_k > z
        k_lo = k_hi - 1
        z_lo, z_hi = self.redshifts[k_lo], self.redshifts[k_hi]
        w = (z - z_lo) / (z_hi - z_lo)
        mu_lo, sd_lo = self._snapshots[k_lo].predict(params)
        mu_hi, sd_hi = self._snapshots[k_hi].predict(params)
        return (1 - w) * mu_lo + w * mu_hi, (1 - w) * sd_lo + w * sd_hi

    def _apply_transform(self, mean, std):
        t = self.meta['output_transform']
        if t == 'log10':
            # emulator trained on 10**y  ->  y = log10(raw);  sigma_y = sigma / (raw ln10)
            std = std / (mean * np.log(10.0))
            mean = np.log10(mean)
        elif t == 'pow10':
            # emulator trained on log10(y)  ->  y = 10**raw;  sigma_y = sigma * y ln10
            mean = 10.0 ** mean
            std = std * mean * np.log(10.0)
        return mean, std

    def predict(self, params, z=0.0, check_bounds=True):
        """
        Predict the statistic for given parameters and redshift.

        Parameters
        ----------
        params : array-like
            Input parameters in scaled units (see ``param_names``):
            shape ``(n_params,)`` for one prediction or ``(n_pred, n_params)``
            for a batch.  7-parameter statistics take
            ``[kappa_w, e_w, M_seed/1e6, v_kin/1e4, eps_kin/1e1, omega_m, sigma_8]``;
            gravity-only statistics take ``[omega_m, sigma_8]``.
        z : float or array-like, optional
            Redshift (default 0).  Must lie within ``z_range``; values between
            trained snapshots are linearly interpolated.  A per-row array of
            length ``n_pred`` is also accepted.
        check_bounds : bool, optional
            Emit a ``RuntimeWarning`` when parameters fall outside the training
            design (default True).

        Returns
        -------
        mean : np.ndarray
            Predicted statistic in physical units on ``x_grid``.
            Shape ``(n_x,)`` for a single input, ``(n_x, n_pred)`` for a batch.
        std : np.ndarray
            GP predictive standard deviation, same shape as ``mean``.
        """
        params = self._check_params(params, check_bounds)
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))

        if z_arr.size == 1:
            mean, std = self._predict_raw_at_z(params, z_arr[0])
        else:
            if z_arr.size != params.shape[0]:
                raise ValueError(
                    f"z has length {z_arr.size} but {params.shape[0]} parameter rows were given"
                )
            mean = np.empty((self.x_grid.size, params.shape[0]))
            std = np.empty_like(mean)
            for zval in np.unique(z_arr):
                rows = np.where(z_arr == zval)[0]
                mean[:, rows], std[:, rows] = self._predict_raw_at_z(params[rows], zval)

        mean, std = self._apply_transform(mean, std)
        if mean.shape[1] == 1:
            return mean[:, 0], std[:, 0]
        return mean, std

    def __repr__(self):
        return (
            f"CosmoHydroEmulator(stat_name='{self.stat_name}', n_params={self.n_params}, "
            f"redshifts={np.round(self.redshifts, 3).tolist()})"
        )


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------
def load_emulator(stat_name, redshifts=None, exp_variance=None):
    """
    Load the emulator for one summary statistic.

    Parameters
    ----------
    stat_name : str
        Name of the summary statistic (see ``list_available_statistics()``).
    redshifts : array-like, optional
        Subset of trained snapshot redshifts to load (default: all).
    exp_variance : float or int, optional
        Override the PCA basis size (default: read from each model file).

    Returns
    -------
    CosmoHydroEmulator

    Examples
    --------
    >>> emu = load_emulator('GSMF')
    >>> params = [3.0, 0.5, 1.0, 0.65, 0.5, 0.14176, 0.8102]
    >>> mean, std = emu.predict(params, z=0.3)
    """
    return CosmoHydroEmulator(stat_name, redshifts=redshifts, exp_variance=exp_variance)


def list_available_statistics():
    """
    List all available summary statistics, grouped by category.

    Returns
    -------
    dict
        ``{'summary': [...], 'profile': [...], 'gravity_only': [...]}``
    """
    return {
        'summary': list(AVAILABLE_STATS_SUMMARY),
        'profile': list(AVAILABLE_STATS_PROFILE),
        'gravity_only': list(AVAILABLE_STATS_GRAVITY_ONLY),
    }
