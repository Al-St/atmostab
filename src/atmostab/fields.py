"""Поля метеовеличин и их тенденции (изменения за 6 часов).

Поля, сопутствующие анализу явлений: относительные геопотенциальные
высоты 500/1000 гПа (толщина слоя), геопотенциальная высота 700 гПа,
приземное давление, температура у земли, ветер на 10 м и 850 гПа, а также
их тенденции между сроками. Все величины в векторизованном виде.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from atmostab import thermo
from atmostab.io import var
from atmostab.indices import P_500, P_700, P_850, at_isobaric
from atmostab.indices import _cape as cape_field


def _clean(da: xr.DataArray, name: str, long_name: str, units: str) -> xr.DataArray:
    da = da.squeeze(drop=True)  # единообразно 2D (lat × lon)
    da = da.rename(name)
    da.attrs.update({"long_name": long_name, "units": units})
    return da


def thickness(ds: xr.Dataset, top_hpa: float = P_500, bottom_hpa: float = 1000) -> xr.DataArray:
    """Относительная геопотенциальная высота (толщина слоя), гпм."""
    h_top = at_isobaric(var(ds, "geopotential_height_isobaric"), top_hpa)
    h_bottom = at_isobaric(var(ds, "geopotential_height_isobaric"), bottom_hpa)
    name = f"H{int(top_hpa)}-{int(bottom_hpa)}"
    return _clean(h_top - h_bottom, name,
                  f"Толщина слоя {int(bottom_hpa)}–{int(top_hpa)} гПа", "gpm")


def geopotential_height(ds: xr.Dataset, level_hpa: float = P_700) -> xr.DataArray:
    """Геопотенциальная высота изобарической поверхности, гпм."""
    h = at_isobaric(var(ds, "geopotential_height_isobaric"), level_hpa)
    return _clean(h, f"HGT{int(level_hpa)}",
                  f"Геопотенциальная высота {int(level_hpa)} гПа", "gpm")


def temperature_2m(ds: xr.Dataset) -> xr.DataArray:
    """Температура воздуха у земли (GFS, ~2 м), °C."""
    t = var(ds, "temperature_2m") - thermo.KELVIN
    return _clean(t, "T2M", "Температура воздуха у земли", "°C")


def surface_pressure(ds: xr.Dataset) -> xr.DataArray:
    """Давление у земли, гПа."""
    p = var(ds, "surface_pressure") / 100.0
    return _clean(p, "PSFC", "Давление у земли", "hPa")


def input_fields(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Штатные поля GFS, используемые как входные данные индексов.

    Возвращаются под именами CAPE, CIN и LI — для отрисовки карт входных
    полей (флаг ``--inputs`` в CLI); сами по себе они не рассчитываются
    пакетом.
    """
    return {
        "CAPE": _clean(var(ds, "cape"), "CAPE",
                       "Доступная энергия конвекции (поле GFS)", "J kg-1"),
        "CIN": _clean(var(ds, "cin"), "CIN",
                      "Энергия сдерживания конвекции (поле GFS)", "J kg-1"),
        "LI": _clean(var(ds, "lifted_index"), "LI",
                     "Surface lifted index (поле GFS)", "K"),
    }


def wind(ds: xr.Dataset, level_hpa: float | None = None) -> tuple[xr.DataArray, xr.DataArray]:
    """Скорость (м/с) и метеорологическое направление (°) ветра.

    Parameters
    ----------
    level_hpa:
        Изобарическая поверхность; ``None`` — приземный ветер.
    """
    if level_hpa is None:
        if "u_wind_10m" in ds and "v_wind_10m" in ds:
            u, v = ds["u_wind_10m"], ds["v_wind_10m"]  # сырые GRIB2 (ветер на 10 м)
        else:
            # в подготовленных NetCDF-файлах приповерхностный ветер
            # представлен полем изобарического набора
            u, v = var(ds, "u_wind_isobaric"), var(ds, "v_wind_isobaric")
        label = "10 м"
        suffix = "10M"
    else:
        u = at_isobaric(var(ds, "u_wind_isobaric"), level_hpa)
        v = at_isobaric(var(ds, "v_wind_isobaric"), level_hpa)
        label = f"{int(level_hpa)} гПа"
        suffix = f"{int(level_hpa)}"
    speed = thermo.wind_speed(u, v)
    direction = thermo.wind_direction(u, v)
    speed = _clean(speed, f"WS{suffix}", f"Скорость ветра ({label})", "m s-1")
    direction = _clean(direction, f"WD{suffix}", f"Направление ветра ({label})", "degree")
    return speed, direction


def tendency(da_now: xr.DataArray, da_before: xr.DataArray,
             hours: float = 6) -> xr.DataArray:
    """Средняя скорость изменения поля (ед./час) между двумя сроками (анализа или прогноза).

    Both fields must be aligned on lat/lon. ВременнЫе координаты (сроки
    анализа различаются по определению) перед вычитанием отбрасываются.
    """
    def _strip(da: xr.DataArray) -> xr.DataArray:
        da = da.squeeze(drop=True)
        return da.drop_vars([c for c in ("time", "valid_time", "step")
                             if c in da.coords])

    now, before = xr.align(_strip(da_now), _strip(da_before), join="exact")
    out = (now - before) / hours
    base_name = str(da_now.name or "field")
    return _clean(out, f"d{base_name}",
                  f"Тенденция: {da_now.attrs.get('long_name', base_name)}",
                  f"{da_now.attrs.get('units', '')} h-1")


def case_tendencies(ds_now: xr.Dataset, ds_before: xr.Dataset,
                    hours: float = 6) -> dict[str, xr.DataArray]:
    """Набор тенденций для анализа эволюции явления.

    Возвращает словарь с тенденциями CAPE, температуры у земли, давления
    у земли и толщины 500–1000 гПа.
    """
    return {
        "dCAPE": tendency(cape_field(ds_now), cape_field(ds_before), hours),
        "dT2M": tendency(temperature_2m(ds_now), temperature_2m(ds_before), hours),
        "dPSFC": tendency(surface_pressure(ds_now), surface_pressure(ds_before), hours),
        "dH500-1000": tendency(thickness(ds_now), thickness(ds_before), hours),
    }
