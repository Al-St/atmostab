"""Скачивание полей GFS 0.25° (прогнозы и архивные анализы) из открытых архивов.

Два источника:

``noaa`` (по умолчанию, анонимно)
    Открытое хранилище NOAA `noaa-gfs-bdp-pds` на AWS S3 (данные примерно
    с конца февраля 2021 г.). Файл анализа ~330 МБ целиком не качается:
    сначала забирается сопутствующий индексный файл ``.idx``
    (перечень сообщений GRIB со смещениями байтов), выбираются только
    нужные сообщения и скачиваются HTTP Range-запросами. Итоговый файл —
    валидный GRIB2 из выбранных сообщений (десятки МБ вместо сотен).

    Внимание: в анализах GFS нет поля спиральности (SRH) и 2-метровых
    полей — они появляются только в прогнозных файлах (f003 и далее).
    Для индекса EHI указывайте ``--forecast-hours 3`` (или другой срок
    кратный трём): будет скачан соответствующий прогнозный файл того же
    запуска. Либо используйте источник ``gdex`` — у RDA спиральность
    есть и в анализах.

``gdex``
    Архив UCAR/RDA `d084001` (NCEP GFS 0.25°, с 15.01.2015):
    <https://gdex.ucar.edu/datasets/d084001/>. Файл
    скачивается целиком (без отбора сообщений). Прямой доступ —
    ``https://data.rda.ucar.edu/d084001/<имя файла>``; сервер UCAR может
    требовать авторизацию (укажите cookie через ``--cookie``) и имеет
    некорректный TLS-сертификат (обходится ``--insecure``).

Использование::

    python -m atmostab.download 2021-05-19 --hours 12 18 --out data/gfs
    python -m atmostab.download 2023-06-15 --hours 0 --forecast-hours 3
    python -m atmostab.download 2016-05-14 --hours 12 --source gdex

Прогресс и итоговые размеры печатаются в stdout; уже скачанные файлы
пропускаются.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import click
import requests

# Бакет NOAA (noaa-gfs-bdp-pds): раскладка ключей менялась.
NOAA_KEY_TEMPLATES = (
    "gfs.{ymd}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.{term}",  # с 23.03.2021
    "gfs.{ymd}/{hour}/gfs.t{hour}z.pgrb2.0p25.{term}",        # до 23.03.2021
)
NOAA_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# UCAR/RDA d084001 — файлы лежат в корне датасета плоским списком.
GDEX_URL = "https://data.rda.ucar.edu/d084001/gfs.0p25{ymdh}.f{forecast:03d}.grib2"

# Изобарические уровни 1000–250 гПа — профиль ветра для BRN.
MB_LEVELS_PROFILE = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
                     600, 550, 500, 450, 400, 350, 300, 275, 250)

# Сообщения GRIB2, необходимые пакету: переменная → множество уровней.
# Токены соответствуют инвентарю wgrib2 (.idx), например
# "257:179005711:d=2021051912:HGT:250 mb:anl:".
IDX_VARIABLES: tuple[tuple[str, frozenset], ...] = tuple(
    (var, frozenset(levels)) for var, levels in (
        ("HGT", {"1000 mb", "850 mb", "700 mb", "500 mb"}),
        ("TMP", {"1000 mb", "850 mb", "700 mb", "500 mb"}),
        ("RH", {"850 mb", "700 mb"}),
        ("SPFH", {"1000 mb", "850 mb", "700 mb", "500 mb"}),
        ("UGRD", {f"{p} mb" for p in MB_LEVELS_PROFILE}),
        ("VGRD", {f"{p} mb" for p in MB_LEVELS_PROFILE}),
        ("CAPE", {"surface"}),
        ("CIN", {"surface"}),
        ("LFTX", {"surface"}),
        ("PRES", {"surface"}),
        # есть только в прогнозных файлах (и в файлах RDA):
        ("HLCY", {"3000-0 m above ground"}),
        ("TMP", {"2 m above ground"}),
        ("UGRD", {"10 m above ground"}),
        ("VGRD", {"10 m above ground"}),
    )
)

# Разобранная строка .idx: смещение, переменная, уровень.
_IDX_LINE = re.compile(r"^\d+:(\d+):d=\d+:(\w+):(.+?):")


def _wanted(line_var: str, line_level: str) -> bool:
    return any(var == line_var and line_level in levels
               for var, levels in IDX_VARIABLES)


def _iter_url(url: str, **kwargs) -> requests.Response:
    response = requests.get(url, stream=True, timeout=120, **kwargs)
    response.raise_for_status()
    return response


def _download_ranges(url: str, ranges: list[tuple[int, int]],
                     chunk: int = 1 << 20) -> bytes:
    """Скачивает перечисленные байтовые диапазоны и склеивает их.

    Каждое сообщение — отдельный Range-запрос; переиспользуется одна
    сессия (keep-alive), чтобы не платить за установку соединения
    на каждое сообщение.
    """
    parts: list[bytes] = []
    with requests.Session() as session:
        for start, end in ranges:
            response = session.get(url, headers={"Range": f"bytes={start}-{end}"},
                                   stream=True, timeout=120)
            response.raise_for_status()
            for block in response.iter_content(chunk_size=chunk):
                parts.append(block)
    return b"".join(parts)


def _select_messages(idx_text: str, total_size: int) -> list[tuple[int, int]]:
    """Выбирает нужные сообщения и возвращает их байтовые диапазоны.

    Диапазон каждого сообщения ограничен началом следующей строки
    инвентаря (выбрана она или нет), поэтому в результат попадает ровно
    то, что запрошено, без промежуточных сообщений.
    """
    lines = []
    for line in idx_text.splitlines():
        match = _IDX_LINE.match(line)
        if not match:
            continue
        lines.append((int(match.group(1)), match.group(2), match.group(3)))

    ranges = []
    for i, (start, var, level) in enumerate(lines):
        if not _wanted(var, level):
            continue
        end = (lines[i + 1][0] - 1) if i + 1 < len(lines) else total_size - 1
        ranges.append((start, end))
    return ranges


def _download_noaa(date: dt.date, hour: int, forecast: int,
                   out_path: Path) -> Path:
    ymd, hour_s = date.strftime("%Y%m%d"), f"{hour:02d}"
    term = "anl" if forecast == 0 else f"f{forecast:03d}"
    if out_path.exists():
        print(f"  {out_path.name}: уже скачан, пропуск")
        return out_path

    idx_url = file_url = None
    idx_text = None
    for template in NOAA_KEY_TEMPLATES:
        key = template.format(ymd=ymd, hour=hour_s, term=term)
        response = requests.get(f"{NOAA_BASE}/{key}.idx", timeout=120)
        if response.ok:
            idx_url, file_url = f"{NOAA_BASE}/{key}.idx", f"{NOAA_BASE}/{key}"
            idx_text = response.text
            break
    if idx_text is None:
        raise RuntimeError(
            f"Файл GFS {ymd} {hour_s}:00 ВСВ (срок {term}) не найден в "
            f"архиве NOAA (данные примерно с конца февраля 2021 г.); "
            f"для более старых дат используйте --source gdex"
        )

    total_size = int(requests.head(file_url, timeout=120,
                                   allow_redirects=True).headers["Content-Length"])
    ranges = _select_messages(idx_text, total_size)
    if not ranges:
        raise RuntimeError("В индексном файле не найдено нужных сообщений")

    print(f"  {file_url.split('amazonaws.com/')[1]}: {len(ranges)} сообщений "
          f"из {len(idx_text.splitlines())}")
    data = _download_ranges(file_url, ranges)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"  → {out_path.name}: {len(data) / 1e6:.1f} МБ")
    return out_path


def _download_gdex(date: dt.date, hour: int, forecast: int,
                   out_path: Path, cookie: str | None, insecure: bool) -> Path:
    if out_path.exists():
        print(f"  {out_path.name}: уже скачан, пропуск")
        return out_path
    url = GDEX_URL.format(ymdh=f"{date:%Y%m%d}{hour:02d}", forecast=forecast)
    headers = {"Cookie": cookie} if cookie else {}
    print(f"  {url}")
    response = _iter_url(url, headers=headers, verify=not insecure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with out_path.open("wb") as f:
        for block in response.iter_content(chunk_size=1 << 20):
            f.write(block)
            size += len(block)
    print(f"  → {out_path.name}: {size / 1e6:.1f} МБ")
    return out_path


def download_gfs(dates: list[dt.date], hours: tuple[int, ...] = (12,),
                 forecast_hours: tuple[int, ...] = (0,), out_dir: str = "data/gfs",
                 source: str = "noaa", cookie: str | None = None,
                 insecure: bool = False) -> list[Path]:
    """Скачивает файлы анализа/прогноза GFS 0.25° за перечисленные сроки.

    Parameters
    ----------
    dates:
        Даты запусков модели.
    hours:
        Сроки запусков (ВСВ), обычно 0, 6, 12, 18.
    forecast_hours:
        Заблаговременности: 0 — анализ, 3, 6, ... — прогнозы.
    out_dir:
        Каталог для файлов (именование: ``gfs_YYYYMMDD_HH_f###.grib2``).
    source:
        ``noaa`` — открытый архив NOAA с отбором сообщений;
        ``gdex`` — архив UCAR/RDA d084001 целиком.
    cookie, insecure:
        Параметры доступа к data.rda.ucar.edu (см. докстринг модуля).

    Returns
    -------
    list[Path]
        Пути к скачанным (или уже существовавшим) GRIB2-файлам.
    """
    out_root = Path(out_dir)
    downloaded: list[Path] = []
    for date in dates:
        for hour in hours:
            for forecast in forecast_hours:
                if forecast and (forecast % 3 or (forecast > 120 and forecast % 12)):
                    print(f"  внимание: срок f{forecast:03d} вне штатной сетки GFS "
                          f"(шаг 3 ч до f120, далее 12 ч до f384) — файл может "
                          f"отсутствовать")
                name = f"gfs_{date:%Y%m%d}_{hour:02d}_f{forecast:03d}.grib2"
                print(f"{date} {hour:02d}:00 ВСВ, срок f{forecast:03d} [{source}]")
                if source == "noaa":
                    downloaded.append(
                        _download_noaa(date, hour, forecast, out_root / name))
                elif source == "gdex":
                    downloaded.append(
                        _download_gdex(date, hour, forecast, out_root / name,
                                       cookie, insecure))
                else:
                    raise ValueError(f"Неизвестный источник: {source!r}")
    return downloaded


@click.command(help="Скачивает поля анализа/прогноза GFS 0.25° для расчёта индексов.")
@click.argument("dates", nargs=-1, required=True,
                help="Даты в формате ГГГГ-ММ-ДД")
@click.option("--hours", default="12", show_default=True,
              help="Сроки запусков через запятую (ВСВ): например 0,6,12,18")
@click.option("--forecast-hours", default="0", show_default=True,
              help="Заблаговременности через запятую; 0 — анализ. "
                   "Для EHI нужен прогнозный срок (3, 6, ...): в анализах "
                   "нет поля спиральности.")
@click.option("--out", "out_dir", default="data/gfs", show_default=True,
              type=click.Path(file_okay=False), help="Каталог для GRIB2-файлов.")
@click.option("--source", type=click.Choice(["noaa", "gdex"]), default="noaa",
              show_default=True, help="Источник данных.")
@click.option("--cookie", default=None, help="Cookie для data.rda.ucar.edu (источник gdex).")
@click.option("--insecure", is_flag=True,
              help="Не проверять TLS-сертификат сервера UCAR (он самоподписанный).")
def main(dates: tuple[str], hours: str, forecast_hours: str, out_dir: str,
         source: str, cookie: str | None, insecure: bool) -> None:
    date_list = [dt.date.fromisoformat(d) for d in dates]
    hour_list = tuple(int(h) for h in hours.split(","))
    forecast_list = tuple(int(f) for f in forecast_hours.split(","))
    try:
        download_gfs(date_list, hour_list, forecast_list, out_dir,
                     source, cookie, insecure)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
