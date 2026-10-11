"""Расчёт индексов неустойчивости атмосферы по полям данных GFS.

Каждый индекс — отдельная функция, принимающая открытый датасет
(:func:`atmostab.open_gfs`) и возвращающая :class:`xarray.DataArray`
с атрибутами ``long_name`` и ``units``; никаких групповых вычислений
«всё сразу» нет — вызывающий код явно перечисляет нужные индексы.
Формулы — по первоисточникам (Miller, 1972; George, 1960; Barlow, 1993),
указанным в docstring функций.

Список индексов
---------------
``KI`` (число Вайтинга), ``TT`` (Total Totals), ``TI`` (индекс
Томпсона), ``SWEAT``, ``KO``, ``DCI``, ``EHI``, ``BRN``, ``UVV``
(скорость конвективного updraft).

CAPE, CIN и Lifted Index пакет не рассчитывает — это **штатные поля
анализа GFS** (``CAPE_L1``, ``CIN_L1``, ``LFT_X_L1`` в подготовленных
NetCDF-файлах; shortName ``CAPE``/``CIN``/``LFTX`` в сырых GRIB2), они
используются как входные данные для TI, DCI, EHI, BRN и UVV.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from atmostab import thermo
from atmostab.io import var

# Уровни, на которых используются стандартные изобарические величины.
P_1000, P_850, P_700, P_500 = 1000, 850, 700, 500
# Слой осреднения ветра для BRN: 1000–250 гПа.
BRN_TOP_HPA = 250


def at_isobaric(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    """Значения поля на изобарической поверхности ``level_hpa`` (гПа).

    Уровень выбирается по значению изобарической координаты,
    а не по позиции в массиве.
    """
    if "isobaric" not in da.dims:
        raise ValueError(f"Поле {da.name!r} не содержит изобарической координаты")
    out = da.sel(isobaric=level_hpa)
    return out.squeeze(drop=True)


def _clean(da: xr.DataArray, name: str, long_name: str, units: str,
           extra: dict | None = None) -> xr.DataArray:
    da = da.squeeze(drop=True)  # единообразно 2D (lat × lon)
    da = da.rename(name)
    da.attrs.update({"long_name": long_name, "units": units})
    if extra:
        da.attrs.update(extra)
    return da


def _t_celsius(ds: xr.Dataset, level_hpa: float) -> xr.DataArray:
    """Температура (°C) на изобарической поверхности."""
    return at_isobaric(var(ds, "temperature_isobaric"), level_hpa) - thermo.KELVIN


def _dewpoint_celsius(ds: xr.Dataset, level_hpa: float) -> xr.DataArray:
    """Точка росы (°C) на изобарической поверхности."""
    t = at_isobaric(var(ds, "temperature_isobaric"), level_hpa)
    rh = at_isobaric(var(ds, "relative_humidity_isobaric"), level_hpa)
    return thermo.dewpoint_from_rh(t, rh)


def _cape(ds: xr.Dataset) -> xr.DataArray:
    """CAPE (Дж/кг) — штатное поле GFS, вход для EHI/BRN/UVV."""
    return var(ds, "cape")


def _lifted_index(ds: xr.Dataset) -> xr.DataArray:
    """Lifted index (K) — штатное поле GFS, вход для TI и DCI."""
    return var(ds, "lifted_index")


def total_totals(ds: xr.Dataset) -> xr.DataArray:
    """TT = T850 + Td850 − 2·T500 (°C).

    Градации (Miller, 1972): TT ≥ 44 — грозы вероятны; ≥ 50 — сильные
    грозы более вероятны; ≥ 55 — очень сильные грозы.
    """
    tt = _t_celsius(ds, P_850) + _dewpoint_celsius(ds, P_850) - 2 * _t_celsius(ds, P_500)
    return _clean(tt, "TT", "Total Totals index", "°C")


def k_index(ds: xr.Dataset) -> xr.DataArray:
    """Ki = T850 − T500 + Td850 − (T700 − Td700) (°C).

    Вероятность гроз по градациям Ki (George, 1960): 0–20 — 0–20 %,
    21–25 — 20–40 %, 26–30 — 40–60 %, 31–35 — 60–80 %, 36–40 — 80–90 %,
    > 40 — более 90 %.
    """
    ki = (_t_celsius(ds, P_850) - _t_celsius(ds, P_500)
          + _dewpoint_celsius(ds, P_850) - (_t_celsius(ds, P_700) - _dewpoint_celsius(ds, P_700)))
    return _clean(ki, "KI", "K-index (число Вайтинга)", "°C")


def thompson_index(ds: xr.Dataset) -> xr.DataArray:
    """Ti = Ki − Li (индекс Томпсона); Ti ≈ 20.4 — ~20 % гроз, Ti ≈ 32.1 — ~55 %."""
    ti = k_index(ds) - _lifted_index(ds)
    return _clean(ti, "TI", "Индекс Томпсона (Ki − Li)", "°C")


def dci(ds: xr.Dataset) -> xr.Dataset:
    """DCI = T850 + Td850 − Li (индекс глубокой конвекции, Barlow 1993); DCI > 30 — глубокая конвекция."""
    d = _t_celsius(ds, P_850) + _dewpoint_celsius(ds, P_850) - _lifted_index(ds)
    return _clean(d, "DCI", "Deep Convective Index", "°C")


def swet(ds: xr.Dataset) -> xr.DataArray:
    """SWEAT — индекс угрозы суровой погоды (Miller, 1972).

    SWEAT = 12·Td850 + 20·max(TT − 49, 0) + 2·V850 + V500 + 125·(S + 0.2),
    где скорости в узлах, S = sin(D500 − D850). Слагаемое 125·(S + 0.2)
    учитывается только если выполняются все условия:

    * 130° ≤ D850 ≤ 250°;
    * 210° ≤ D500 ≤ 310°;
    * D500 > D850;
    * V850 ≥ 15 узлов и V500 ≥ 15 узлов.

    Слагаемое 125·(S + 0.2) добавляется только при выполнении всех
    четырёх перечисленных условий. Итог обрезается снизу нулём.
    """
    td850 = _dewpoint_celsius(ds, P_850)
    tt = total_totals(ds)

    u850 = at_isobaric(var(ds, "u_wind_isobaric"), P_850)
    v850 = at_isobaric(var(ds, "v_wind_isobaric"), P_850)
    u500 = at_isobaric(var(ds, "u_wind_isobaric"), P_500)
    v500 = at_isobaric(var(ds, "v_wind_isobaric"), P_500)

    spd850 = thermo.ms_to_knots(thermo.wind_speed(u850, v850))
    spd500 = thermo.ms_to_knots(thermo.wind_speed(u500, v500))
    dir850 = thermo.wind_direction(u850, v850)
    dir500 = thermo.wind_direction(u500, v500)

    base = 12 * td850 + np.maximum(0, 20 * (tt - 49)) + 2 * spd850 + spd500
    shear_gate = (
        (dir850 >= 130) & (dir850 <= 250)
        & (dir500 >= 210) & (dir500 <= 310)
        & (dir500 > dir850)
        & (spd850 >= 15) & (spd500 >= 15)
    )
    shear_term = xr.where(
        shear_gate, 125 * (np.sin(np.radians(dir500 - dir850)) + 0.2), 0.0
    )
    out = np.maximum(base + shear_term, 0.0)
    return _clean(out, "SWEAT", "Индекс угрозы суровой погоды (SWEAT)",
                  "points", {"thresholds": ">300 — грозы, >400 — смерчи"})


def ko_index(ds: xr.Dataset) -> xr.DataArray:
    """KO = (θe500 + θe700 − θe850 − θe1000) / 2 (K); KO ≤ 4 — неустойчивость.

    θe — эквивалентно-потенциальная температура (см. :func:`atmostab.thermo.theta_e`).
    """
    q = var(ds, "specific_humidity_isobaric")
    thetas = []
    for level in (P_500, P_700, P_850, P_1000):
        thetas.append(thermo.theta_e(
            at_isobaric(var(ds, "temperature_isobaric"), level),
            at_isobaric(q, level),
            level,
        ))
    ko = (thetas[0] + thetas[1] - thetas[2] - thetas[3]) / 2
    return _clean(ko, "KO", "KO-index (потенциальная неустойчивость слоя 850–700 гПа)", "K",
                  {"thresholds": "<=4 — неустойчивость"})


def ehi(ds: xr.Dataset) -> xr.DataArray:
    """EHI = CAPE·SRH / 160000 (м²/с²·Дж/кг); EHI > 1 — суперячейки и торнадо возможны.

    SRH — спиральность слоя 0–3 км из полей GFS (``HLCY_Y103``).
    """
    out = _cape(ds) * var(ds, "storm_relative_helicity") / 160000.0
    return _clean(out, "EHI", "Energy Helicity Index", "dimensionless",
                  {"thresholds": ">1 — суперячейки/торнадо"})


def brn(ds: xr.Dataset) -> xr.DataArray:
    """Массовое число Ричардсона: BRN = CAPE / (0.5·(ū² + v̄²)).

    ū и v̄ — осреднённые по слою 1000–250 гПа модули составляющих ветра.
    BRN < 45 — конвекция при значительном сдвиге ветра (условия суперячеек).
    """
    u = var(ds, "u_wind_isobaric")
    v = var(ds, "v_wind_isobaric")
    # Профиль ветра в слое 1000–250 гПа (19 стандартных уровней GFS).
    levels_desc = np.sort(u["isobaric"].values)[::-1]
    levels = levels_desc[levels_desc >= BRN_TOP_HPA][:19]
    profile_u = u.sel(isobaric=levels)
    profile_v = v.sel(isobaric=levels)
    u_mean = np.abs(profile_u).mean("isobaric")
    v_mean = np.abs(profile_v).mean("isobaric")
    out = _cape(ds) / (0.5 * (u_mean ** 2 + v_mean ** 2))
    return _clean(out, "BRN", "Bulk Richardson Number (по средним модулям ветра 1000–250 гПа)",
                  "dimensionless", {"thresholds": "<45 — условия суперячеек"})


def updraft_velocity(ds: xr.Dataset) -> xr.DataArray:
    """UVV = √(2·CAPE) — оценка максимальной скорости конвективного updraft (м/с)."""
    out = np.sqrt(2 * _cape(ds))
    return _clean(out, "UVV", "Вертикальная скорость конвективного потока √(2·CAPE)", "m s-1")


#: Реестр индексов пакета: имя → функция расчёта. Используется CLI
#: (:mod:`atmostab.cli`) для вызова индексов по имени из ``--index``;
#: в коде предпочтительны прямые вызовы функций.
COMPUTED_INDICES = {
    "KI": k_index,
    "TT": total_totals,
    "TI": thompson_index,
    "SWEAT": swet,
    "KO": ko_index,
    "DCI": dci,
    "EHI": ehi,
    "BRN": brn,
    "UVV": updraft_velocity,
}
