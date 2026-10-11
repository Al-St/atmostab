"""Тесты термодинамических функций."""

from __future__ import annotations

import numpy as np
import xarray as xr

from atmostab import thermo


def test_dewpoint_of_saturated_air_equals_temperature():
    t = xr.DataArray([273.15, 283.15, 293.15])
    td = thermo.dewpoint_from_rh(t, xr.DataArray([100.0, 100.0, 100.0]))
    np.testing.assert_allclose(td.values, t.values - 273.15, atol=1e-6)


def test_dewpoint_reference_value():
    # T = 25 °C, RH = 50 % → Td ≈ 13.9 °C (таблицы Магнуса)
    t = xr.DataArray([298.15])
    td = thermo.dewpoint_from_rh(t, xr.DataArray([50.0]))
    assert abs(float(td.values[0]) - 13.86) < 0.15


def test_theta_e_increases_with_humidity():
    t = xr.DataArray([280.0])
    p = 850.0
    dry = float(thermo.theta_e(t, xr.DataArray([0.0]), p).values[0])
    moist = float(thermo.theta_e(t, xr.DataArray([0.01]), p).values[0])
    assert moist > dry
    # 10 г/кг добавляют ~25 K скрытой теплоты
    assert 20 < moist - dry < 30


def test_wind_direction_meteorological_convention():
    # v < 0 (движение к югу), u = 0 → ветер северный (дует с севера): 0°
    assert abs(float(thermo.wind_direction(xr.DataArray(0.0), xr.DataArray(-10.0)))) < 1e-6
    # u < 0 (движение к западу), v = 0 → ветер восточный: 90°
    assert abs(float(thermo.wind_direction(xr.DataArray(-10.0), xr.DataArray(0.0))) - 90) < 1e-6
    # u > 0, v = 0 → ветер западный: 270°
    assert abs(float(thermo.wind_direction(xr.DataArray(10.0), xr.DataArray(0.0))) - 270) < 1e-6


def test_ms_to_knots():
    # 15 м/с ≈ 29.2 узла
    kt = thermo.ms_to_knots(xr.DataArray(15.0))
    assert abs(float(kt) - 29.17) < 0.05
