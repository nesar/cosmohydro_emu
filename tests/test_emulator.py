"""
Tests for the emulator module.
"""

import warnings

import numpy as np
import pytest

from cosmohydro_emu.emulator import get_data_path
from cosmohydro_emu import (
    AVAILABLE_STATS,
    AVAILABLE_STATS_GRAVITY_ONLY,
    AVAILABLE_STATS_PROFILE,
    AVAILABLE_STATS_SUMMARY,
    FIDUCIAL_COSMOLOGY,
    get_redshifts,
    get_x_grid,
    list_available_statistics,
    load_emulator,
)

PARAMS_7P = [3.0, 0.5, 1.0, 0.65, 0.5, FIDUCIAL_COSMOLOGY['omega_m'], FIDUCIAL_COSMOLOGY['sigma_8']]
PARAMS_COSMO = [FIDUCIAL_COSMOLOGY['omega_m'], FIDUCIAL_COSMOLOGY['sigma_8']]


def params_for(emu):
    return PARAMS_7P if emu.n_params == 7 else PARAMS_COSMO


# Loading a full multi-z emulator takes ~0.3 s; share instances across tests.
@pytest.fixture(scope="module")
def emulators():
    return {stat: load_emulator(stat) for stat in AVAILABLE_STATS}


class TestEmulatorLoading:
    """Test emulator loading functionality."""

    def test_load_7p_emulator(self, emulators):
        emu = emulators['GSMF']
        assert emu.stat_name == 'GSMF'
        assert emu.n_params == 7
        assert emu.n_redshifts == 11

    def test_load_cosmo_only_emulator(self, emulators):
        emu = emulators['Pk_GO']
        assert emu.n_params == 2
        assert emu.param_keys == ['omega_m', 'sigma_8']

    def test_invalid_stat_name(self):
        with pytest.raises(ValueError):
            load_emulator('INVALID_STAT')

    def test_list_available_statistics(self):
        stats = list_available_statistics()
        assert set(stats) == {'summary', 'profile', 'gravity_only'}
        assert stats['summary'] == AVAILABLE_STATS_SUMMARY
        assert stats['profile'] == AVAILABLE_STATS_PROFILE
        assert stats['gravity_only'] == AVAILABLE_STATS_GRAVITY_ONLY
        assert len(AVAILABLE_STATS) == 14

    def test_redshifts_sorted_and_consistent(self, emulators):
        for stat, emu in emulators.items():
            assert np.all(np.diff(emu.redshifts) > 0)
            assert np.allclose(emu.redshifts, get_redshifts(stat))
            assert emu.z_range == (emu.redshifts[0], emu.redshifts[-1])
            assert emu.z_range[0] == pytest.approx(0.0, abs=1e-6)

    def test_load_subset_of_redshifts(self):
        emu = load_emulator('GSMF', redshifts=[0.0, 1.0])
        assert emu.n_redshifts == 2
        assert np.allclose(emu.redshifts, [0.0, 0.9996], atol=1e-3)

    def test_n_pc_read_from_pickle(self, emulators):
        # PCA basis size must come from the saved model, not a default
        assert emulators['Pk-ratio'].n_pc == [1, 1, 1, 1, 1]
        assert emulators['Pk_GO'].n_pc == [2, 2, 2, 2, 2]


class TestEmulatorPredictions:
    """Test emulator prediction functionality."""

    def test_single_prediction_shape(self, emulators):
        emu = emulators['GSMF']
        x_grid, _ = get_x_grid('GSMF')
        mean, std = emu.predict(PARAMS_7P)
        assert mean.shape == (x_grid.size,)
        assert std.shape == mean.shape

    def test_batch_prediction_shape(self, emulators):
        emu = emulators['HMF']
        params = np.array([PARAMS_7P] * 4)
        mean, std = emu.predict(params, z=0.0)
        assert mean.shape == (emu.x_grid.size, 4)
        assert std.shape == mean.shape
        # batch equals stacked single predictions
        m1, _ = emu.predict(PARAMS_7P)
        assert np.allclose(mean[:, 0], m1)

    def test_large_batch_chunking(self, emulators):
        emu = emulators['fGas']
        params = np.array([PARAMS_7P] * 150)
        mean, std = emu.predict(params, z=0.3)
        assert mean.shape == (emu.x_grid.size, 150)
        assert np.allclose(mean, mean[:, :1])

    def test_prediction_values_finite_and_std_positive(self, emulators):
        mean, std = emulators['CGD'].predict(PARAMS_7P, z=0.25)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std > 0)

    def test_wrong_param_count(self, emulators):
        with pytest.raises(ValueError):
            emulators['GSMF'].predict(PARAMS_7P[:5])
        with pytest.raises(ValueError):
            emulators['Pk_GO'].predict(PARAMS_7P)

    def test_out_of_range_parameters_warn(self, emulators):
        bad = list(PARAMS_7P)
        bad[0] = 10.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            emulators['GSMF'].predict(bad)
        assert any(issubclass(x.category, RuntimeWarning) and 'outside' in str(x.message) for x in w)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            emulators['GSMF'].predict(bad, check_bounds=False)
        assert not any('outside' in str(x.message) for x in w)

    def test_reproduces_training_data(self, emulators):
        """At a training design point the emulator should return ~the training curve."""
        emu = emulators['Pk-ratio']
        d = np.load(get_data_path('Pk-ratio'))
        k0 = int(np.argmin(np.abs(d['redshifts'] - 0.0)))
        truth = d['y_vals'][:3, k0, :]
        mean, _ = emu.predict(d['p_train'][:3], z=0.0)
        assert np.max(np.abs(mean.T - truth)) < 0.03

    def test_output_transforms(self, emulators):
        # GSMF/HMF return dn/dlog10M (small positive numbers), not the 10** training values
        mean, _ = emulators['GSMF'].predict(PARAMS_7P)
        assert mean.max() < 0.1 and mean.max() > 0
        # Pk_GO returns P(k) in (Mpc/h)^3, i.e. large positive numbers
        mean, _ = emulators['Pk_GO'].predict(PARAMS_COSMO)
        assert mean.max() > 1e3 and np.all(mean > 0)


class TestRedshiftHandling:
    """Test redshift interpolation between snapshots."""

    def test_at_snapshot_equals_direct(self, emulators):
        emu = emulators['GSMF']
        z1 = emu.redshifts[3]
        m_a, s_a = emu.predict(PARAMS_7P, z=z1)
        m_b, s_b = emu.predict(PARAMS_7P, z=z1 + 1e-6)
        assert np.allclose(m_a, m_b) and np.allclose(s_a, s_b)

    def test_linear_interpolation(self, emulators):
        emu = emulators['fGas']  # no output transform -> exactly linear
        z_lo, z_hi = emu.redshifts[1], emu.redshifts[2]
        m_lo, _ = emu.predict(PARAMS_7P, z=z_lo)
        m_hi, _ = emu.predict(PARAMS_7P, z=z_hi)
        m_mid, _ = emu.predict(PARAMS_7P, z=0.5 * (z_lo + z_hi))
        assert np.allclose(m_mid, 0.5 * (m_lo + m_hi))

    def test_per_row_redshifts(self, emulators):
        emu = emulators['HMF']
        params = np.array([PARAMS_7P] * 3)
        zs = [0.0, 0.5, 1.5]
        mean, _ = emu.predict(params, z=zs)
        for i, z in enumerate(zs):
            m_single, _ = emu.predict(PARAMS_7P, z=z)
            assert np.allclose(mean[:, i], m_single)

    def test_redshift_out_of_range(self, emulators):
        with pytest.raises(ValueError):
            emulators['GSMF'].predict(PARAMS_7P, z=3.0)
        with pytest.raises(ValueError):
            emulators['CGD'].predict(PARAMS_7P, z=1.0)   # profiles only trained to z=0.5
        with pytest.raises(ValueError):
            emulators['CSFR'].predict(PARAMS_7P, z=0.5)  # z=0 only
        with pytest.raises(ValueError):
            emulators['GSMF'].predict(PARAMS_7P, z=-0.1)

    def test_mismatched_z_length(self, emulators):
        with pytest.raises(ValueError):
            emulators['GSMF'].predict(np.array([PARAMS_7P] * 3), z=[0.0, 1.0])


class TestAllEmulators:
    """Every shipped emulator loads and predicts at every trained snapshot."""

    @pytest.mark.parametrize("stat_name", AVAILABLE_STATS)
    def test_all_snapshots(self, emulators, stat_name):
        emu = emulators[stat_name]
        x_grid, _ = get_x_grid(stat_name)
        for z in emu.redshifts:
            mean, std = emu.predict(params_for(emu), z=z)
            assert mean.shape == (x_grid.size,)
            assert np.all(np.isfinite(mean))
            assert np.all(np.isfinite(std))
            assert np.all(std >= 0)

    @pytest.mark.parametrize("stat_name", AVAILABLE_STATS)
    def test_repr(self, emulators, stat_name):
        assert stat_name in repr(emulators[stat_name])
