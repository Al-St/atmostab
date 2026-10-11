"""Тесты отображения: реестр фиксированных шкал покрывает все поля."""

from atmostab import indices
from atmostab.fields import input_fields
from atmostab.plotting import DEFAULT_RANGES


def test_default_ranges_cover_all_fields():
    fields = set(indices.COMPUTED_INDICES) | set(input_fields.__name__ and {"CAPE", "CIN", "LI"})
    missing = fields - set(DEFAULT_RANGES)
    assert not missing, f"нет фиксированной шкалы для: {sorted(missing)}"


def test_ranges_are_sane():
    for name, (vmin, vmax) in DEFAULT_RANGES.items():
        assert vmin < vmax, name
