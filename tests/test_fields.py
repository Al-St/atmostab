"""Тесты полей метеовеличин и тенденций."""

from __future__ import annotations

import numpy as np

from atmostab import fields


def test_thickness(synthetic_ds):
    th = fields.thickness(synthetic_ds)
    # Толщина 1000–500 гПа при T~5..288K должна быть порядка 5 км
    assert 4500 < float(th.isel(lat=0, lon=0)) < 6000


def test_surface_pressure_units(synthetic_ds):
    p = fields.surface_pressure(synthetic_ds)
    np.testing.assert_allclose(p.values, 995.0)


def test_tendency_6h(synthetic_ds):
    now = synthetic_ds.copy(deep=True)
    now["CAPE_L1"].values = synthetic_ds.CAPE_L1.values + 600.0
    tend = fields.tendency(fields.temperature_2m(now),
                           fields.temperature_2m(synthetic_ds), hours=6)
    np.testing.assert_allclose(tend.values, np.zeros_like(tend.values))

    from atmostab.indices import _cape as cape_field
    cape_tend = fields.tendency(cape_field(now), cape_field(synthetic_ds), hours=6)
    np.testing.assert_allclose(cape_tend.values, 100.0)  # 600 Дж/кг за 6 ч


def test_case_tendencies_keys(synthetic_ds):
    later = synthetic_ds.copy(deep=True)
    later["TMP_L1"].values += 3.0
    tend = fields.case_tendencies(later, synthetic_ds)
    assert set(tend) == {"dCAPE", "dT2M", "dPSFC", "dH500-1000"}
    np.testing.assert_allclose(tend["dT2M"].values, 0.5)  # 3 K за 6 часов


def test_input_fields_keys_and_shape(synthetic_ds):
    inputs = fields.input_fields(synthetic_ds)
    assert set(inputs) == {"CAPE", "CIN", "LI"}
    expected = synthetic_ds.CAPE_L1.isel(time=0).shape
    for name, da in inputs.items():
        assert da.shape == expected, name
        assert np.isfinite(da.values).all(), name
