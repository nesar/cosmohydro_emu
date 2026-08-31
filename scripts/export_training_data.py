"""One-time export of trained models + training data into the package tree.

This script populates ``cosmohydro_emu/models/<STAT>/`` (SEPIA pickles) and
``cosmohydro_emu/data/<STAT>_training_data.npz`` (the exact training arrays
needed to rebuild each model's PCA basis at load time) from the CosmoHydro
project directory.  It is *not* needed to use the package -- the exported
files ship with it.  Re-run it only after retraining emulators.

Per statistic the ``.npz`` contains

    p_train        (n_train, n_params)      scaled design parameters
    y_vals         (n_train, n_z, n_y)      training targets, trained snapshots only
    y_ind          (n_y,)                   x-grid (mass / radius / k / scale factor)
    z_index_range  (n_z,)                   index used in the pickle file name
    redshifts      (n_z,)                   redshift of each trained snapshot
    snapshot_ids   (n_z,)                   HACC snapshot number (or -1 if n/a)
    param_names    (n_params,)              plain-text parameter names

Snapshots are stored in the order of ``z_index_range``; the package sorts by
redshift itself.

Usage (from the repository root):
    python scripts/export_training_data.py [--cosmohydro /path/to/CosmoHydro]
"""

import argparse
import os
import shutil

import numpy as np
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(HERE, '..', 'cosmohydro_emu')
DATA_OUT = os.path.join(PKG_DIR, 'data')
MODEL_OUT = os.path.join(PKG_DIR, 'models')

SEED_MASS_SCALE, VKIN_SCALE, EPS_SCALE = 1e6, 1e4, 1e1
PARAM_NAMES = np.array(['kappa_w', 'e_w', 'M_seed/1e6', 'v_kin/1e4',
                        'eps_kin/1e1', 'omega_m', 'sigma_8'])
COSMO_COLS = [5, 6]
TRAIN_IDX = np.arange(100)            # runs 000-099; 100-109 held out

MULTIZ_STATS = ['GSMF', 'HMF', 'fGas',
                'CGD', 'CGED', 'CPP', 'CTP', 'CEP', 'CEEP', 'CMP', 'CYP']

# P(k) models live at 5 redshift tags; we index them 0..4 in *descending* z
# to match the HACC snapshot convention used by the multi-z models.
PK_ZTAGS = ['2.0', '1.0', '0.5', '0.1', '0.0']
PK_KMIN, PK_KMAX = 0.015707963267948967, 8.042477193189871   # 2pi/L .. Nyquist


def fill_nan_with_interpolation(data, kind):
    """Identical to cosmo_hydro_emu.load_hacc.fill_nan_with_interpolation."""
    out = np.copy(data)
    for i in range(data.shape[0]):
        bad = np.where(np.isnan(data[i]) | (data[i] < 1e-6))[0]
        good = np.where(~np.isnan(data[i]) & (data[i] > 1e-6))[0]
        if len(bad) > 0 and len(good) > 0:
            f = interp1d(good, data[i][good], kind=kind, fill_value='extrapolate')
            out[i][bad] = f(bad)
    return out


def load_design(cosmohydro):
    p = np.loadtxt(os.path.join(cosmohydro, 'data', 'FinalDesign.txt'),
                   delimiter=',', skiprows=1)[:110].copy()
    p[:, 2] /= SEED_MASS_SCALE
    p[:, 3] /= VKIN_SCALE
    p[:, 4] /= EPS_SCALE
    return p


def save_npz(stat, **payload):
    path = os.path.join(DATA_OUT, f'{stat}_training_data.npz')
    np.savez_compressed(path, **payload)
    y = payload['y_vals']
    assert not np.isnan(y).any(), f'{stat}: NaNs in exported training data'
    print(f'  {stat:6s} data   -> {os.path.relpath(path)}  y_vals {y.shape}  '
          f'z = {np.round(payload["redshifts"], 3).tolist()}')


def copy_model(src, stat, z_index):
    dst_dir = os.path.join(MODEL_OUT, stat)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f'multivariate_model_z_index{z_index}.pkl')
    shutil.copyfile(src, dst)
    return dst


# ---------------------------------------------------------------------------
def export_multiz(cosmohydro):
    for stat in MULTIZ_STATS:
        src_dir = os.path.join(cosmohydro, 'models', f'{stat}_multiz')
        d = np.load(os.path.join(src_dir, 'training_data.npz'))
        zi = d['z_index_range']
        for z in zi:
            copy_model(os.path.join(src_dir, f'multivariate_model_z_index{z}.pkl'), stat, int(z))
        save_npz(stat,
                 p_train=d['p_train'],
                 y_vals=d['y_vals'][:, zi, :],
                 y_ind=d['y_ind'],
                 z_index_range=zi,
                 redshifts=d['redshifts'][zi],
                 snapshot_ids=d['snapshot_ids'][zi],
                 param_names=PARAM_NAMES)


def export_csfr(cosmohydro):
    extract = os.path.join(cosmohydro, 'data',
                           'scidac-400MPC_RUNS_5SG_2COSMO_PARAM-extracts_20260323')
    arr = None
    for i in range(110):
        c = np.loadtxt(os.path.join(extract, f'RUN{i:03d}', 'extract', 'CSFR.txt'))
        if arr is None:
            arr = np.zeros((110, c.shape[0]))
        arr[i] = c[:, 1]
    a = c[:, 0]
    arr = fill_nan_with_interpolation(arr, 'linear')
    cond = np.where((a >= 0.0) & (a <= 1.0))[0]
    copy_model(os.path.join(cosmohydro, 'models', 'CSFR_multivariate_model_z_index0.pkl'),
               'CSFR', 0)
    save_npz('CSFR',
             p_train=load_design(cosmohydro)[TRAIN_IDX],
             y_vals=arr[TRAIN_IDX][:, None, cond],
             y_ind=a[cond],
             z_index_range=np.array([0]),
             redshifts=np.array([0.0]),
             snapshot_ids=np.array([624]),
             param_names=PARAM_NAMES)


def _load_pk(pk_dir, ztag):
    P = Pgo = k_ref = None
    for i in range(110):
        h = np.loadtxt(os.path.join(pk_dir, f'run{i:03d}_z{ztag}.hydro.full.pk.txt'))
        g = np.loadtxt(os.path.join(pk_dir, f'run{i:03d}_z{ztag}.go.pk.txt'))
        if P is None:
            k_ref = h[:, 0]
            P = np.zeros((110, k_ref.size))
            Pgo = np.zeros((110, k_ref.size))
        assert np.allclose(h[:, 0], k_ref) and np.allclose(g[:, 0], k_ref)
        P[i], Pgo[i] = h[:, 1], g[:, 1]
    m = (k_ref > PK_KMIN) & (k_ref < PK_KMAX)
    return k_ref[m], P[:, m], Pgo[:, m]


def export_pk(cosmohydro):
    pk_dir = os.path.join(cosmohydro, 'data', 'scidac-olcf-pk_3')
    design = load_design(cosmohydro)
    ratio_y, go_y, k_ref, zs = [], [], None, []
    for zi, ztag in enumerate(PK_ZTAGS):
        k, P, Pgo = _load_pk(pk_dir, ztag)
        if k_ref is None:
            k_ref = k
        assert np.allclose(k, k_ref), f'k grid differs at z={ztag}'
        ratio_y.append((P / Pgo)[TRAIN_IDX])
        go_y.append(np.log10(Pgo[TRAIN_IDX]))
        zs.append(float(ztag))
        if ztag == '0.0':
            src = os.path.join(cosmohydro, 'models', 'Pk_multivariate_model_z_index0.pkl')
        else:
            src = os.path.join(cosmohydro, 'models', 'Pk_cosmo', f'ratio_z{ztag}.pkl')
        copy_model(src, 'Pk', zi)
        copy_model(os.path.join(cosmohydro, 'models', 'Pk_cosmo', f'logP_go_z{ztag}.pkl'),
                   'Pk_GO', zi)

    # consistency with the notebook-exported z=0 training arrays
    ref = np.load(os.path.join(cosmohydro, 'models', 'Pk_training_data.npz'))
    assert np.allclose(ref['y_ind'], k_ref) and np.allclose(ref['y_vals'], ratio_y[-1])

    common = dict(y_ind=k_ref, z_index_range=np.arange(len(PK_ZTAGS)),
                  redshifts=np.array(zs), snapshot_ids=np.full(len(PK_ZTAGS), -1))
    save_npz('Pk', p_train=design[TRAIN_IDX], y_vals=np.stack(ratio_y, axis=1),
             param_names=PARAM_NAMES, **common)
    save_npz('Pk_GO', p_train=design[TRAIN_IDX][:, COSMO_COLS], y_vals=np.stack(go_y, axis=1),
             param_names=PARAM_NAMES[COSMO_COLS], **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cosmohydro', default=os.path.join(HERE, '..', '..', 'CosmoHydro'),
                    help='path to the CosmoHydro project directory')
    args = ap.parse_args()
    os.makedirs(DATA_OUT, exist_ok=True)
    os.makedirs(MODEL_OUT, exist_ok=True)
    print('Exporting from', os.path.abspath(args.cosmohydro))
    export_multiz(args.cosmohydro)
    export_csfr(args.cosmohydro)
    export_pk(args.cosmohydro)
    print('done.')


if __name__ == '__main__':
    main()
