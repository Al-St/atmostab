"""CLI: расчёт индексов неустойчивости и построение карт.

Индексы обязательно перечисляются в ``--index`` — повторением флага
или через запятую, регистр не важен::

    atmostab gfs_20261011_06_f012.grib2 --index SWEAT --index EHI
    atmostab gfs_20261011_06_f012.grib2 --index sweat,ehi
    atmostab data/gfs/*.grib2 --index KI,TT,SWEAT,EHI --tendency
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

from atmostab import plotting
from atmostab.io import open_gfs
from atmostab.indices import COMPUTED_INDICES

logger = logging.getLogger("atmostab")

# Ключевые индексы для основной панели 2×2 и остальные — для дополнительной.
CORE_PANEL = ["KI", "TT", "SWEAT", "EHI"]
EXTRA_PANEL = ["TI", "DCI", "KO", "BRN", "UVV"]


def parse_indices(values: tuple[str, ...]) -> tuple[str, ...]:
    """Разбирает значения ``--index``: повторения флага и/или запятые.

    Регистр не важен, повторы схлопываются. Неизвестные имена — ошибка
    со списком доступных.
    """
    names = tuple(dict.fromkeys(
        name.strip().upper()
        for value in values
        for name in value.split(",")
        if name.strip()
    ))
    unknown = [n for n in names if n not in COMPUTED_INDICES]
    if unknown:
        raise click.BadParameter(
            f"неизвестные индексы: {', '.join(unknown)}. "
            f"Доступны: {', '.join(COMPUTED_INDICES)}"
        )
    return names


@click.command(help="Рассчитывает перечисленные в --index индексы неустойчивости "
                    "для файлов данных GFS (анализ или прогноз, NetCDF или сырые "
                    "GRIB2) и строит карты.")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output-dir", default="figures", show_default=True,
              type=click.Path(file_okay=False), help="Каталог для PNG-карт.")
@click.option("--index", "indices", required=True, multiple=True,
              help="Индексы к расчёту (обязательно): повторением флага или "
                   "через запятую, например --index SWEAT,EHI (регистр не важен). "
                   "Доступны: KI, TT, TI, SWEAT, KO, DCI, EHI, BRN, UVV.")
@click.option("--dpi", default=150, show_default=True, help="Разрешение сохраняемых карт.")
@click.option("--region", default=None,
              help="Вырезка области «запад,восток,юг,север» в градусах "
                   "(полезно для глобальных GRIB2: без неё индексы считаются "
                   "по всему глобу).")
@click.option("--inputs/--no-inputs", "with_inputs", default=False,
              help="Дополнительно построить карты входных полей GFS: "
                   "CAPE, CIN, Lifted index.")
@click.option("--tendency/--no-tendency", "with_tendency", default=False,
              help="Для пары сроков построить карты тенденций CAPE/T/давления/толщины.")
def main(files: tuple[str], output_dir: str, indices: tuple[str], dpi: int,
         region: str | None, with_inputs: bool, with_tendency: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    region_box = _parse_region(region)
    names = parse_indices(indices)

    datasets = [(path, open_gfs(path, region=region_box)) for path in files]

    for path, ds in datasets:
        stamp = _time_stamp(ds)
        suptitle = f"Индексы неустойчивости, {stamp}"

        # Если у какого-то индекса нет входных полей (например, SRH для EHI
        # в файлах анализа), он пропускается с предупреждением — остальные
        # считаются.
        selected: dict[str, object] = {}
        for name in names:
            try:
                selected[name] = COMPUTED_INDICES[name](ds)
            except KeyError as exc:
                logger.warning("индекс %s пропущен: %s", name, exc.args[0])
        if not selected:
            raise click.ClickException(
                f"ни один из индексов ({', '.join(names)}) не удалось рассчитать "
                f"— в файле нет необходимых полей"
            )

        core = {name: selected[name] for name in CORE_PANEL if name in selected}
        if core:
            out_path = out_dir / f"{Path(path).stem}_core.png"
            plotting.panel(core, suptitle=suptitle, path=str(out_path), dpi=dpi)
            logger.info("%s → %s", stamp, out_path)

        extra = {name: selected[name] for name in EXTRA_PANEL if name in selected}
        if extra:
            extra_path = out_dir / f"{Path(path).stem}_extra.png"
            plotting.panel(extra, suptitle=suptitle, path=str(extra_path), dpi=dpi)
            logger.info("%s → %s", stamp, extra_path)

        if with_inputs:
            from atmostab import fields
            inputs = fields.input_fields(ds)
            inputs_path = out_dir / f"{Path(path).stem}_inputs.png"
            # CIN ≤ 0: Blues_r — ноль белым, сильный запирающий слой
            # тёмно-синим (после фиксированной шкалы −500…0); LI —
            # дивергентная палитра относительно нуля.
            plotting.panel(inputs, suptitle=f"Входные поля GFS, {stamp}",
                           path=str(inputs_path), dpi=dpi,
                           cmaps={"LI": "RdBu_r", "CIN": "Blues_r"})
            logger.info("входные поля %s → %s", stamp, inputs_path)

    if with_tendency:
        if len(datasets) < 2:
            logger.warning("--tendency: нужен минимум два срока в FILES — "
                           "тенденции не строятся")
        else:
            _plot_tendencies(datasets, out_dir, dpi, region_box)


def _parse_region(region: str | None) -> tuple[float, float, float, float] | None:
    if region is None:
        return None
    values = [float(v) for v in region.split(",")]
    if len(values) != 4:
        raise click.BadParameter("Ожидается «запад,восток,юг,север», например 15,55,38,53")
    return tuple(values)


def _time_stamp(ds) -> str:
    """Срок действия данных: для прогнозов — valid_time, иначе время запуска."""
    for coord in ("valid_time", "time"):
        if coord in ds.coords:
            values = np.asarray(ds[coord].values).ravel()
            if values.size:
                return str(values[0])[:16].replace("T", " ")
    return "срок неизвестен"


def _valid_time(ds) -> np.datetime64 | None:
    for coord in ("valid_time", "time"):
        if coord in ds.coords:
            values = np.asarray(ds[coord].values).ravel()
            if values.size:
                return values[0].astype("datetime64[s]")
    return None


def _plot_tendencies(datasets, out_dir: Path, dpi: int,
                     region_box) -> None:
    from atmostab import fields

    for (path_now, ds_now), (_, ds_before) in zip(datasets[1:], datasets[:-1]):
        if not (ds_now.sizes["lat"] == ds_before.sizes["lat"]
                and ds_now.sizes["lon"] == ds_before.sizes["lon"]
                and float(ds_now["lon"].min()) == float(ds_before["lon"].min())
                and float(ds_now["lat"].min()) == float(ds_before["lat"].min())):
            logger.warning("пропуск тенденций %s: сетка не совпадает с предыдущим файлом",
                           Path(path_now).name)
            continue

        t_now, t_before = _valid_time(ds_now), _valid_time(ds_before)
        if t_now is None or t_before is None or t_now <= t_before:
            logger.warning("пропуск тенденций %s: не определить интервал между сроками",
                           Path(path_now).name)
            continue
        hours = float((t_now - t_before) / np.timedelta64(1, "s")) / 3600.0

        tend = fields.case_tendencies(ds_now, ds_before, hours=hours)
        stamp = _time_stamp(ds_now)
        out_path = out_dir / f"{Path(path_now).stem}_tendency.png"
        plotting.panel(tend, suptitle=f"Тенденции за {hours:g} ч, к {stamp}",
                       path=str(out_path), dpi=dpi,
                       cmap="RdBu_r",
                       cmaps={"dCAPE": "RdBu_r"})
        logger.info("тенденции %s → %s", stamp, out_path)


if __name__ == "__main__":
    main()
