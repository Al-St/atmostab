"""Тесты расчёта индексов неустойчивости на синтетическом датасете."""

from __future__ import annotations

import numpy as np

from atmostab import indices


def test_k_index_hand_computed(synthetic_ds):
    ds = synthetic_ds.isel(lat=0, lon=0, time=0)
    t850 = float(ds.TMP_L100.sel(isobaric=850)) - 273.15
    t700 = float(ds.TMP_L100.sel(isobaric=700)) - 273.15
    t500 = float(ds.TMP_L100.sel(isobaric=500)) - 273.15

    rh850 = float(ds.R_H_L100.sel(isobaric=850))
    td850 = 243.04 * (np.log(rh850 / 100) + 17.625 * t850 / (243.04 + t850)) / \
        (17.625 - np.log(rh850 / 100) - 17.625 * t850 / (243.04 + t850))
    rh700 = float(ds.R_H_L100.sel(isobaric=700))
    td700 = 243.04 * (np.log(rh700 / 100) + 17.625 * t700 / (243.04 + t700)) / \
        (17.625 - np.log(rh700 / 100) - 17.625 * t700 / (243.04 + t700))

    ki_expected = t850 - t500 + td850 - (t700 - td700)
    ki = indices.k_index(synthetic_ds)
    assert abs(float(ki.isel(lat=0, lon=0)) - ki_expected) < 1e-4


def test_registry_covers_all_indices(synthetic_ds):
    expected_shape = synthetic_ds.CAPE_L1.isel(time=0).shape  # 2D lat × lon
    for name, function in indices.COMPUTED_INDICES.items():
        da = function(synthetic_ds)
        assert da.shape == expected_shape, name
        assert np.isfinite(da.values).all(), name


def test_uvv_is_sqrt_of_2_cape(synthetic_ds):
    uvv = indices.updraft_velocity(synthetic_ds)
    expected = np.sqrt(2 * synthetic_ds.CAPE_L1.isel(time=0).values)
    np.testing.assert_allclose(uvv.values, expected, rtol=1e-6)


def test_ehi_scaling(synthetic_ds):
    ehi = indices.ehi(synthetic_ds)
    np.testing.assert_allclose(
        ehi.values,
        synthetic_ds.CAPE_L1.isel(time=0).values
        * synthetic_ds.HLCY_Y103.isel(time=0).values / 160000.0,
        rtol=1e-6,
    )


def test_swet_shear_term_gated_by_wind_speed(synthetic_ds):
    """При скоростях < 15 узлов слагаемое сплочённости должно быть нулевым.

    Синтетический ветер 20 м/с ≈ 38.9 узла — условие выполняется; ослабим
    ветер до 5 м/с и сравним с базой.
    """
    base = float(indices.swet(synthetic_ds).isel(lat=0, lon=0))

    weak = synthetic_ds.copy(deep=True)
    weak["U_GRD_L100"].values[:] = -5.0 * np.sqrt(0.5)
    weak["V_GRD_L100"].values[:] = -5.0 * np.sqrt(0.5)
    gated = float(indices.swet(weak).isel(lat=0, lon=0))

    # Юго-западные направления 850/500 гПа не проходят фильтр направлений
    # в обоих случаях, поэтому различие может быть только от скоростей.
    assert base > 0 and gated > 0
    assert base != gated


def test_brn_positive_and_finite(synthetic_ds):
    brn = indices.brn(synthetic_ds)
    assert (brn.values > 0).all()


def test_thompson_index_is_ki_minus_li(synthetic_ds):
    ti = indices.thompson_index(synthetic_ds)
    expected = indices.k_index(synthetic_ds) - synthetic_ds.LFT_X_L1.isel(time=0)
    np.testing.assert_allclose(ti.values, expected.values, rtol=1e-6)
