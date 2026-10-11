"""Базовые термодинамические функции и утилиты для работы с ветром.

Все функции векторизованы (numpy/xarray) и принимают на вход
:class:`xarray.DataArray` либо :class:`numpy.ndarray`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

KELVIN = 273.15

# Коэффициенты формулы Магнуса (Alduchov & Eskridge, 1996).
MAGNUS_A = 17.625
MAGNUS_B = 243.04  # K

# Теплота парообразования (Дж/кг) и удельная теплоёмкость воздуха (Дж/(кг·К)),
# используемые в аппроксимации эквивалентно-потенциальной температуры.
LATENT_HEAT = 2.5e6
CP_DRY_AIR = 1004.0


def dewpoint_from_rh(temperature: xr.DataArray, rh_percent: xr.DataArray) -> xr.DataArray:
    """Температура точки росы (°C) по температуре (K) и влажности (%).

    Инверсия формулы Магнуса (Alduchov & Eskridge, 1996)::

        Td = B * g / (A - g),   g = ln(RH/100) + A*(T-273.15)/(B + T-273.15)

    Влажность обрезается снизу 0.1 %: глобальные поля GFS содержат
    RH = 0, при которых формула расходится.
    """
    t_c = temperature - KELVIN
    rh = np.clip(rh_percent, 0.1, 100.0)
    gamma = np.log(rh / 100) + MAGNUS_A * t_c / (MAGNUS_B + t_c)
    return MAGNUS_B * gamma / (MAGNUS_A - gamma)


def theta_e(temperature: xr.DataArray, specific_humidity: xr.DataArray,
            pressure_hpa: xr.DataArray | float) -> xr.DataArray:
    """Аппроксимация эквивалентно-потенциальной температуры (K)::

        θe ≈ (T + L·q/cp) · (1000/p)^0.286

    где T — температура (K), q — удельная влажность (кг/кг), p — давление (гПа).
    """
    theta = (temperature + LATENT_HEAT * specific_humidity / CP_DRY_AIR) \
        * (1000.0 / pressure_hpa) ** 0.286
    theta.attrs["units"] = "K"
    theta.attrs["long_name"] = "Эквивалентно-потенциальная температура (аппроксимация)"
    return theta


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Модуль скорости ветра (м/с) по составляющим u (запад→восток) и v (юг→север)."""
    spd = np.sqrt(u ** 2 + v ** 2)
    spd.attrs["units"] = "m s-1"
    spd.attrs["long_name"] = "Скорость ветра"
    return spd


def wind_direction(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Метеорологическое направление ветра (°), т.е. откуда дует: 0° — север, 90° — восток.

    Вычисляется как ``(180/π)·atan2(-u, -v) mod 360``,
    где u — зональная, v — меридиональная составляющая (м/с).
    """
    deg = np.degrees(np.arctan2(-u, -v)) % 360
    deg.attrs["units"] = "degree"
    deg.attrs["long_name"] = "Направление ветра (метеорологическое)"
    return deg


def ms_to_knots(speed: xr.DataArray) -> xr.DataArray:
    """Перевод скорости ветра из м/с в узлы."""
    kt = speed * (3600.0 / 1852.0)
    kt.attrs = {**speed.attrs, "units": "kt"}
    return kt
