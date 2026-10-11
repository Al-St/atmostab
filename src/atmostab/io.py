"""Чтение и валидация файлов данных GFS (анализов и прогнозов).

Поддерживаются два формата (тип определяется автоматически):

1. **Сырые GRIB2** — файлы, скачиваемые из открытых архивов NOAA/UCAR
   (:mod:`atmostab.download`), читаются через cfgrib. Канонические ключи
   из :data:`VAR_NAMES` отображаются на имена переменных GRIB
   (``t``, ``r``, ``q``, ``u``, ``v``, ``gh``, ``cape``, ``cin``,
   ``lftx``, ``pres``, ``srh``) по типам уровней.
2. **Подготовленные NetCDF** — региональные вырезки анализа GFS
   (декодер с суффиксами: ``TMP_L100`` — изобарические поверхности с
   координатой ``level3`` в мбар, ``_L1`` — у поверхности и т.д.;
   описание см. в ``data/README.md``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import xarray as xr

# Канонические имена переменных пакета → представления в разных форматах.
VAR_NAMES: dict[str, str] = {
    "temperature_isobaric": "TMP_L100",        # температура, K
    "relative_humidity_isobaric": "R_H_L100",  # относительная влажность, %
    "specific_humidity_isobaric": "SPF_H_L100",  # удельная влажность, кг/кг
    "u_wind_isobaric": "U_GRD_L100",           # зональная составляющая ветра, м/с
    "v_wind_isobaric": "V_GRD_L100",           # меридиональная составляющая ветра, м/с
    "geopotential_height_isobaric": "HGT_L100",  # геопотенциальная высота, гпм
    "cape": "CAPE_L1",                         # CAPE (приземная частица), Дж/кг
    "cin": "CIN_L1",                           # CIN, Дж/кг
    "lifted_index": "LFT_X_L1",                # индекс lifted index, K
    "storm_relative_helicity": "HLCY_Y103",    # спиральность 0–3 км, м²/с²
    "temperature_2m": "TMP_L1",                # температура воздуха у земли, K
    "surface_pressure": "PRES_L1",             # давление у земли, Па
}

# Соответствие канонических ключей переменным GRIB2 (shortName eccodes)
# по типам уровней. Замечание: давление у поверхности в eccodes — 'sp',
# а температура встречается и на изобарах ('t'), и у земли ('t', 2 м).
GRIB_ISOBARIC = {
    "temperature_isobaric": "t",
    "relative_humidity_isobaric": "r",
    "specific_humidity_isobaric": "q",
    "u_wind_isobaric": "u",
    "v_wind_isobaric": "v",
    "geopotential_height_isobaric": "gh",
}
GRIB_SURFACE = {
    "cape": "cape",
    "cin": "cin",
    "lifted_index": "lftx",
    "surface_pressure": "sp",
}
# (shortName, высота над землёй в метрах)
GRIB_HEIGHT = {
    "temperature_2m": ("t", 2.0),
    "u_wind_10m": ("u", 10.0),
    "v_wind_10m": ("v", 10.0),
}
# Спиральность — слой 3000–0 м (typeOfLevel heightAboveGroundLayer,
# shortName hlcy): отдельная группа без координаты высоты.
GRIB_LAYER = {
    "storm_relative_helicity": "hlcy",
}

# Минимально необходимый набор переменных для расчёта индексов
# (спиральность нужна только EHI, её отсутствие обрабатывается мягко).
REQUIRED_VARS = (
    "temperature_isobaric",
    "relative_humidity_isobaric",
    "u_wind_isobaric",
    "v_wind_isobaric",
    "cape",
    "lifted_index",
)

GRIB_SUFFIXES = {".grib", ".grib2", ".grb", ".grb2"}


def var(ds: xr.Dataset, key: str) -> xr.DataArray:
    """Возвращает переменную датасета по каноническому ключу из :data:`VAR_NAMES`.

    Raises
    ------
    KeyError
        Если переменная отсутствует в файле.
    """
    if key in ds:          # датасет из GRIB2: переменные уже названы канонически
        return ds[key]
    name = VAR_NAMES[key]  # датасет из NetCDF
    if name not in ds:
        raise KeyError(
            f"В файле нет переменной {name!r} ({key}); "
            f"доступные переменные: {sorted(map(str, ds.data_vars))}"
        )
    return ds[name]


def open_gfs(path, region: "tuple[float, float, float, float] | None" = None) -> xr.Dataset:
    """Открывает файл данных GFS (анализ или прогноз) и валидирует его.

    Для NetCDF координата изобарических поверхностей (``level3``)
    переводится из мбар в гПа и переименовывается в ``isobaric``.
    Для GRIB2 группы сообщений (изобарические поля, приповерхностные
    величины) сливаются в один датасет с теми же каноническими именами.

    Parameters
    ----------
    path:
        Путь к файлу (``.grib2`` читается через cfgrib, остальное —
        как NetCDF).
    region:
        Необязательная вырезка области ``(запад, восток, юг, север)`` в
        градусах; позволяет не считать индексы по всему глобу — сырые
        файлы GFS глобальные (721×1440).
    """
    path = Path(path)
    if path.suffix.lower() in GRIB_SUFFIXES:
        ds = _open_grib(path)
    else:
        ds = _open_netcdf(path)

    ds = _standardize_coords(ds)

    if region is not None:
        west, east, south, north = region
        lat_desc = bool(ds["lat"].values[0] > ds["lat"].values[-1])
        lat_slice = slice(north, south) if lat_desc else slice(south, north)
        ds = ds.sel(lon=slice(west, east), lat=lat_slice)
        if ds.sizes.get("lat", 0) == 0 or ds.sizes.get("lon", 0) == 0:
            raise ValueError(f"Область {region} не пересекается с сеткой файла")

    missing = [k for k in REQUIRED_VARS
               if VAR_NAMES[k] not in ds and k not in ds]
    if missing:
        raise KeyError(
            f"В {path} отсутствуют обязательные поля: "
            f"{[VAR_NAMES[k] for k in missing]}"
        )
    return ds


def _standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Приводит имена координат к пакeтным: lat, lon, isobaric (гПа)."""
    renames = {c: n for c, n in (("latitude", "lat"), ("longitude", "lon"))
               if c in ds.coords}
    if renames:
        ds = ds.rename(renames)
    for coord in ("level3", "isobaric", "level"):
        if coord in ds.coords and coord != "isobaric":
            ds = ds.rename({coord: "isobaric"})
            break
    if "isobaric" in ds.coords:
        ds["isobaric"].attrs["units"] = "hPa"
    return ds


def _open_netcdf(path) -> xr.Dataset:
    return xr.open_dataset(path)


def _open_grib(path) -> xr.Dataset:
    """Читает GRIB2 через cfgrib и сводит группы сообщений в один датасет.

    cfgrib разбивает файл на группы сообщений; u/v на изобарических
    уровнях GFS декодируются в отдельную группу, поэтому изобарические
    группы объединяются. Ожидается один срок анализа/прогноза на файл.
    """
    import cfgrib  # локальный импорт: зависимость нужна только для GRIB2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # indexpath="" — не создавать рядом с файлом служебные .idx-файлы
        # (и не натыкаться на устаревшие после замены файла).
        groups = cfgrib.open_datasets(str(path),
                                      backend_kwargs={"errors": "ignore",
                                                      "indexpath": ""})

    parts = []
    for group in groups:
        if "isobaricInhPa" in group.coords:
            parts.append(_rename_group(group, GRIB_ISOBARIC)
                         .rename({"isobaricInhPa": "isobaric"}))
        elif "surface" in group.coords:
            parts.append(_rename_group(group, GRIB_SURFACE))
        elif "heightAboveGround" in group.coords:
            parts.append(_select_height_levels(group))
        elif "heightAboveGroundLayer" in group.coords:
            parts.append(_rename_group(group, GRIB_LAYER))

    isobaric = [p for p in parts if "isobaric" in p.dims or "isobaric" in p.coords]
    if not isobaric:
        raise ValueError(f"В {path} не найдены поля на изобарических поверхностях")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # join="outer": у разных переменных свои наборы изобарических
        # уровней (например, u/v — профиль 1000–250 гПа, T — 4 уровня).
        ds = xr.merge(parts, join="outer", compat="override")
    return ds


def _rename_group(group: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """Оставляет из группы только нужные переменные под каноническими именами."""
    renames = {short: key for key, short in mapping.items() if short in group}
    out = group[list(renames)].rename(renames)
    return _first_term(out)


def _select_height_levels(group: xr.Dataset) -> xr.Dataset:
    """Выбирает величины на заданных высотах над землёй (2 м, 10 м, 3000 м)."""
    pieces = []
    heights = group["heightAboveGround"].values
    for key, (short, level) in GRIB_HEIGHT.items():
        if short in group and level in heights:
            pieces.append(group[[short]].sel(heightAboveGround=level)
                          .rename({short: key})
                          .drop_vars("heightAboveGround"))
    return xr.merge(_first_term(p) for p in pieces) if pieces else xr.Dataset()


def _first_term(group: xr.Dataset) -> xr.Dataset:
    """Оставляет один срок/шаг прогноза (ожидается один срок на файл)."""
    for coord in ("step", "time", "valid_time"):
        if coord in group.dims and group.sizes[coord] > 1:
            group = group.isel({coord: 0})
        elif coord in group.coords and group[coord].size > 1:
            group = group.isel({coord: 0})
    return group
