"""
Tests for the data_utils module.
"""

import numpy as np
import pytest

from cosmohydro_emu import (
    AVAILABLE_STATS,
    AVAILABLE_STATS_GRAVITY_ONLY,
    FIDUCIAL_COSMOLOGY,
    PARAM_KEYS,
    get_parameter_info,
    get_plot_info,
    get_redshifts,
    get_statistic_info,
    get_valid_range,
    get_x_grid,
)


class TestDataUtilities:
    """Test data utility functions."""

    @pytest.mark.parametrize("stat_name", AVAILABLE_STATS)
    def test_get_x_grid(self, stat_name):
        x_grid, x_label = get_x_grid(stat_name)
        assert isinstance(x_grid, np.ndarray)
        assert len(x_grid) > 0
        assert np.all(np.diff(x_grid) > 0)
        assert isinstance(x_label, str) and len(x_label) > 0

    @pytest.mark.parametrize("stat_name", AVAILABLE_STATS)
    def test_get_redshifts(self, stat_name):
        z = get_redshifts(stat_name)
        assert isinstance(z, np.ndarray)
        assert len(z) >= 1
        assert np.all(np.diff(z) > 0)
        assert z[0] == pytest.approx(0.0, abs=1e-6)

    def test_redshift_coverage(self):
        assert len(get_redshifts('GSMF')) == 11
        assert len(get_redshifts('HMF')) == 11
        assert len(get_redshifts('fGas')) == 7
        assert len(get_redshifts('CGD')) == 5
        assert len(get_redshifts('Pk-ratio')) == 5
        assert len(get_redshifts('CSFR')) == 1
        assert get_redshifts('GSMF').max() == pytest.approx(2.0, abs=0.01)
        assert get_redshifts('CGD').max() == pytest.approx(0.5, abs=0.01)

    @pytest.mark.parametrize("stat_name", AVAILABLE_STATS)
    def test_get_plot_info(self, stat_name):
        plot_info = get_plot_info(stat_name)
        for key in ('title', 'xlabel', 'ylabel', 'xscale', 'yscale'):
            assert key in plot_info
        assert plot_info['xscale'] in ['linear', 'log']
        assert plot_info['yscale'] in ['linear', 'log']

    @pytest.mark.parametrize("stat_name", AVAILABLE_STATS)
    def test_get_valid_range(self, stat_name):
        valid_range = get_valid_range(stat_name)
        assert isinstance(valid_range, tuple)
        assert len(valid_range) == 2
        assert valid_range[0] < valid_range[1]
        # the recommended range must overlap the emulator's x-grid
        x_grid, _ = get_x_grid(stat_name)
        assert valid_range[0] < x_grid.max()
        assert valid_range[1] > x_grid.min()

    def test_get_parameter_info(self):
        info = get_parameter_info()
        for key in ('names', 'latex_names', 'ranges', 'descriptions', 'scales', 'fiducial'):
            assert key in info
        assert info['names'] == PARAM_KEYS
        assert len(info['names']) == 7
        assert len(info['ranges']) == 7
        assert len(info['descriptions']) == 7
        assert len(info['scales']) == 3          # M_seed, v_kin, epsilon_kin
        assert info['fiducial'] == FIDUCIAL_COSMOLOGY
        for lo, hi in info['ranges'].values():
            assert lo < hi

    def test_get_parameter_info_per_statistic(self):
        assert get_parameter_info('GSMF')['names'] == PARAM_KEYS
        for stat in AVAILABLE_STATS_GRAVITY_ONLY:
            assert get_parameter_info(stat)['names'] == ['omega_m', 'sigma_8']
            assert get_parameter_info(stat)['scales'] == {}

    def test_get_statistic_info(self):
        info = get_statistic_info()
        assert set(info) == set(AVAILABLE_STATS)
        assert info['Pk_GO']['n_params'] == 2
        assert info['GSMF']['n_params'] == 7
        assert info['GSMF']['n_x'] == len(get_x_grid('GSMF')[0])
        single = get_statistic_info('CGD')
        assert list(single) == ['CGD'] and single['CGD']['category'] == 'profile'

    def test_invalid_stat_name(self):
        for fn in (get_x_grid, get_redshifts, get_plot_info, get_valid_range,
                   get_statistic_info, get_parameter_info):
            with pytest.raises(ValueError):
                fn('INVALID_STAT')
