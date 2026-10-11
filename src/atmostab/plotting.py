"""Визуализация полей индексов и метеовеличин на картах (cartopy).

Карты строятся как заливки изолиний (``contourf``) на географической
проекции с береговыми линиями 50 м; одиночные карты — в ортографической
проекции, панели — в PlateCarree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Регион анализа по умолчанию: восток Европейской части России и Казахстан.
DEFAULT_EXTENT = (15, 55, 38, 53)   # (запад, восток, юг, север), градусы
PROJECTION_CENTER = (35, 43)        # (долгота, широта) центра проекции


# Палитры для знакопеременных полей (тенденций): нулевое значение = нейтральный цвет.
DIVERGING_CMAPS = {"RdBu_r", "RdBu", "seismic", "bwr", "coolwarm", "BrBG"}

# Фиксированные шкалы отображения индексов и входных полей, гПа/пороги —
# как в оперативных продуктах. Без них шкала подстраивается под экстремумы
# всего домена, и сигнал в зоне явления тонет (например, у BRN максимум
# вдали от бури в разы больше значений в зоне бури). Значения вне
# диапазона уходят в цвета концов шкалы (extend="both").
DEFAULT_RANGES = {
    "KI": (-20, 40),      # вероятность гроз: значимая градация 26–40+
    "TT": (26, 60),       # ≥44 грозы, ≥50 сильные
    "TI": (-10, 40),      # ~20 % гроз при 20.4, ~55 % при 32.1
    "SWEAT": (0, 500),    # >300 грозы, >400 смерчи
    "KO": (-10, 12),      # ≤4 — неустойчивость
    "DCI": (-10, 35),     # >30 — глубокая конвекция
    "EHI": (-0.5, 2),     # >1 — суперячейки/торнадо
    "BRN": (0, 100),      # <45 — условия суперячеек
    "UVV": (0, 60),       # м/с, √(2·CAPE)
    "CAPE": (0, 2500),    # ≳2500 — мощная конвекция
    "CIN": (-500, 0),     # ≤0 (соглашение GFS)
    "LI": (-10, 10),      # <0 — неустойчивость
}


def _figure_axes(ncols: int, nrows: int, subplot_kws: dict,
                 data_aspect: float = 15.0 / 40.0):
    """Создаёт сетку осей с высотой ряда, подобранной под аспект карт.

    Панели строятся в PlateCarree: у ортографической проекции фиксированный
    аспект, из-за которого любое изменение позиции axes (colorbar,
    компоновщики) непредсказуемо смещает карты; прямоугольные ячейки
    ведут себя детерминированно. ``data_aspect`` — отношение широты к
    долготе области; высота ряда = высота карты + место под colorbar и
    заголовок, иначе между рядами остаются большие пустые полосы.
    """
    figw = 6.4 * ncols
    cell_w = figw * 0.79 / (ncols + 0.30 * (ncols - 1))  # с учётом wspace
    row_h = data_aspect * cell_w + 0.45  # карта + подписи осей + заголовок
    figh = row_h * nrows + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw, figh),
                             subplot_kw=subplot_kws, squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.85, top=0.88, bottom=0.07,
                        wspace=0.30, hspace=0.45)
    for ax in axes.ravel():
        ax.set_anchor("N")  # карты и colorbar прижаты к верху ячейки
        # Ключевая настройка GeoAxes: по умолчанию аспект подгоняется
        # изменением пределов данных ('datalim'), из-за чего colorbar
        # и компоновщики непредсказуемо смещают карту; 'box' меняет рамку.
        ax.set_adjustable("box")
    return fig, axes


def _symmetric_limits(da: xr.DataArray) -> tuple[float, float]:
    """Симметричные относительно нуля пределы для дивергентной палитры."""
    finite = da.values[np.isfinite(da.values)]
    bound = float(np.abs(finite).max()) if finite.size else 1.0
    return -bound, bound


def _data_extent(da: xr.DataArray) -> tuple[float, float, float, float]:
    """Границы области (з, в, ю, с) из координат поля."""
    return (float(da["lon"].min()), float(da["lon"].max()),
            float(da["lat"].min()), float(da["lat"].max()))


def field_map(da: xr.DataArray, ax=None, cmap: str = "magma_r",
              title: str | None = None, extent: Sequence[float] | None = None,
              vmin=None, vmax=None, levels: int | Sequence[float] = 20,
              left_labels: bool = True):
    """Строит карту поля значений на географической проекции.

    Если ``extent`` не задан, область берётся из координат поля.
    ``levels`` — число уровней изолиний или явный массив уровней.
    ``left_labels=False`` убирает подписи широт (в панелях — для правой
    колонки, чтобы подписи единиц colorbar соседа не перекрывали их).

    Returns
    -------
    (fig, ax, mesh)
        Объекты matplotlib и результат ``contourf`` (для colorbar).
    """
    if extent is None:
        extent = _data_extent(da)
    if ax is None:
        center = ((extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2)
        fig, ax = plt.subplots(figsize=(11, 8),
                               subplot_kw={"projection": ccrs.Orthographic(*center)})
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    # Шаг меридианов/параллелей подбирается под ширину области, чтобы
    # подписи не слипались на узких доменах.
    lon_span, lat_span = extent[1] - extent[0], extent[3] - extent[2]
    gl.xlocator = plt.MaxNLocator(min(int(lon_span / 8) + 1, 8), steps=[1, 2, 5, 10])
    gl.ylocator = plt.MaxNLocator(min(int(lat_span / 6) + 1, 8), steps=[1, 2, 5, 10])
    ax.coastlines(resolution="50m", linewidth=0.6)

    if isinstance(levels, int) and (vmin is not None or vmax is not None):
        # Явные уровни: с целым levels matplotlib игнорирует vmin/vmax.
        lo = vmin if vmin is not None else float(np.nanmin(da.values))
        hi = vmax if vmax is not None else float(np.nanmax(da.values))
        levels = np.linspace(lo, hi, levels + 1)

    mesh = ax.contourf(da["lon"], da["lat"], da.values,
                       transform=ccrs.PlateCarree(), cmap=cmap,
                       levels=levels, vmin=vmin, vmax=vmax,
                       extend="both" if isinstance(levels, np.ndarray)
                       else "neither")
    if title:
        # ax.text вместо ax.set_title: с gridliner-подписями set_title
        # вытесняется за пределы axes (особенность cartopy GeoAxes).
        ax.text(0.5, 1.12, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11)
    return ax, mesh


def panel(fields: Mapping[str, xr.DataArray], suptitle: str | None = None,
          ncols: int = 2, extent: Sequence[float] | None = None,
          cmap: str = "magma_r", path: str | None = None, dpi: int = 200,
          cmaps: Mapping[str, str] | None = None):
    """Строит панель из нескольких полей.

    Parameters
    ----------
    fields:
        Словарь «заголовок → поле значений» — например, собранный из
        результатов функций :mod:`atmostab.indices` или результат
        :func:`atmostab.fields.case_tendencies`.
    extent:
        Границы карты (запад, восток, юг, север); по умолчанию — из данных.
    path:
        Если задан — сохраняет рисунок в файл и закрывает фигуру.
    """
    n = len(fields)
    nrows = -(-n // ncols)
    cmaps = cmaps or {}
    extent = _data_extent(next(iter(fields.values()))) if extent is None \
        else tuple(extent)
    data_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, axes = _figure_axes(ncols, nrows, {"projection": ccrs.PlateCarree()},
                             data_aspect=data_aspect)
    axes = axes.ravel()

    last_mesh = None
    for i, (ax, (title, da)) in enumerate(zip(axes, fields.items())):
        cmap_i = cmaps.get(title, cmap)
        vmin = vmax = None
        if cmap_i in DIVERGING_CMAPS:
            vmin, vmax = _symmetric_limits(da)
        elif title in DEFAULT_RANGES:
            vmin, vmax = DEFAULT_RANGES[title]
        _, mesh = field_map(da, ax=ax, extent=extent, cmap=cmap_i,
                            title=title, vmin=vmin, vmax=vmax,
                            # подписи широт — только у левой колонки:
                            # иначе подпись единиц colorbar соседа слева
                            # перекрывает их
                            left_labels=(i % ncols == 0))
        # Компактный colorbar-инсет справа вплотную к карте: расположенный
        # ниже карты, он увеличивает зазор между рядами панели.
        cax = ax.inset_axes([1.04, 0.0, 0.035, 1.0])
        cbar = fig.colorbar(mesh, cax=cax)
        units = da.attrs.get("units", "")
        if units:
            cbar.set_label(units, fontsize=9, rotation=270, labelpad=10)
        last_mesh = mesh

    for ax in axes[n:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)

    if path:
        # ВАЖНО: без bbox_inches="tight" — он считает неверный общий bbox
        # по устаревшим экстентам подписей gridliner и обрезает фигуру.
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return fig
