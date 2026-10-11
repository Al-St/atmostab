"""atmostab — расчёт индексов неустойчивости атмосферы по данным GFS.

Пакет является переработанной (векторизованной) версией алгоритмов,
разработанных в выпускной квалификационной работе
«Влияние неустойчивости на динамику атмосферных процессов» (РГГМУ, 2022).

Читает и подготовленные NetCDF-вырезки, и сырые GRIB2-файлы (через
cfgrib); скачать прогнозы можно командой ``atmostab-download``
(см. :mod:`atmostab.download`).

Индексы вычисляются по одному — явно вызванной функцией::

    import atmostab
    from atmostab import indices

    ds = atmostab.open_gfs("gfs_20261011_06_f012.grib2",
                           region=(15, 55, 38, 53))
    ki = indices.k_index(ds)
    sweat = indices.swet(ds)
"""

from atmostab import plotting
from atmostab.io import open_gfs

__version__ = "1.1.0"

__all__ = ["open_gfs", "plotting", "__version__"]
