"""Тесты CLI: разбор значений --index."""

from __future__ import annotations

import click
import pytest

from atmostab.cli import parse_indices
from atmostab.indices import COMPUTED_INDICES


def test_empty_selects_nothing():
    assert parse_indices(()) == ()


def test_repeated_flags():
    assert parse_indices(("SWEAT", "EHI")) == ("SWEAT", "EHI")


def test_comma_separated_and_case_insensitive():
    assert parse_indices(("sweat,ehi",)) == ("SWEAT", "EHI")


def test_mixed_forms_with_duplicates():
    assert parse_indices(("ki", "SWEAT,ki")) == ("KI", "SWEAT")


def test_unknown_index_raises_with_hint():
    with pytest.raises(click.BadParameter) as exc:
        parse_indices(("SWET",))  # опечатка
    message = str(exc.value)
    assert "SWET" in message and "SWEAT" in message
    assert ", ".join(COMPUTED_INDICES) in message
