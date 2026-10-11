"""Общие фикстуры: синтетический датасет в формате подготовленных NetCDF-файлов GFS."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


def make_synthetic_dataset(nlat: int = 4, nlon: int = 5) -> xr.Dataset:
    """Минимальный датасет со структурой региональных вырезок GFS
    (изобарическая координата, суффиксы имён переменных, один срок).

    Температурный профиль — стандартная атмосфера с насыщением на 850 гПа
    и сухим воздухом выше; ветер — постоянный юго-западный 20 м/с.
    """
    lat = np.linspace(38, 53, nlat)
    lon = np.linspace(15, 55, nlon)
    levels = np.array([1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
                       600, 550, 500, 450, 400, 350, 300, 275, 250, 225, 200,
                       175, 150, 100], dtype=float)  # мбар

    def field(levels_factor=lambda p: p):
        return xr.DataArray(
            np.zeros((1, levels.size, nlat, nlon), dtype="float32"),
            dims=("time", "isobaric", "lat", "lon"),
            coords={"time": [np.datetime64("2021-05-19T12:00:00")],
                    "isobaric": levels, "lat": lat, "lon": lon},
        )

    tmp = field()
    # T(p) = 288 - 50*(1 - p/1000)*2: линейный профиль ~6.5 K/км
    tmp.values[:] = (288.0 - 50.0 * 2 * (1 - levels[None, :, None, None] / 1000.0))
    rh = field()
    rh.values[:] = 20.0
    rh.values[:, levels == 850] = 100.0   # насыщение у 850 гПа
    sh = field()
    sh.values[:] = 0.004
    u = field()
    v = field()
    u.values[:] = -20.0 * np.sqrt(0.5)    # юго-западный ветер
    v.values[:] = -20.0 * np.sqrt(0.5)
    hgt = field()
    hgt.values[:] = np.linspace(0, 11000, levels.size)[None, :, None, None]

    ds = xr.Dataset(
        {
            "TMP_L100": tmp.assign_attrs(units="K"),
            "R_H_L100": rh.assign_attrs(units="%"),
            "SPF_H_L100": sh.assign_attrs(units="kg kg-1"),
            "U_GRD_L100": u.assign_attrs(units="m s-1"),
            "V_GRD_L100": v.assign_attrs(units="m s-1"),
            "HGT_L100": hgt.assign_attrs(units="gpm"),
            "CAPE_L1": xr.DataArray(np.full((1, nlat, nlon), 1500.0, dtype="float32"),
                                    dims=("time", "lat", "lon"),
                                    coords={"time": tmp.time, "lat": lat, "lon": lon},
                                    attrs={"units": "J kg-1"}),
            "CIN_L1": xr.DataArray(np.full((1, nlat, nlon), -50.0, dtype="float32"),
                                   dims=("time", "lat", "lon"),
                                   coords={"time": tmp.time, "lat": lat, "lon": lon},
                                   attrs={"units": "J kg-1"}),
            "LFT_X_L1": xr.DataArray(np.full((1, nlat, nlon), -2.0, dtype="float32"),
                                     dims=("time", "lat", "lon"),
                                     coords={"time": tmp.time, "lat": lat, "lon": lon},
                                     attrs={"units": "K"}),
            "HLCY_Y103": xr.DataArray(np.full((1, nlat, nlon), 200.0, dtype="float32"),
                                      dims=("time", "lat", "lon"),
                                      coords={"time": tmp.time, "lat": lat, "lon": lon},
                                      attrs={"units": "m2 s-2"}),
            "TMP_L1": xr.DataArray(np.full((1, nlat, nlon), 291.0, dtype="float32"),
                                   dims=("time", "lat", "lon"),
                                   coords={"time": tmp.time, "lat": lat, "lon": lon},
                                   attrs={"units": "K"}),
            "PRES_L1": xr.DataArray(np.full((1, nlat, nlon), 99500.0, dtype="float32"),
                                    dims=("time", "lat", "lon"),
                                    coords={"time": tmp.time, "lat": lat, "lon": lon},
                                    attrs={"units": "Pa"}),
        }
    )
    return ds


@pytest.fixture
def synthetic_ds() -> xr.Dataset:
    return make_synthetic_dataset()
