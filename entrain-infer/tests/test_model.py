"""Units are a claim too: they are checked before any number in the report is read."""
from __future__ import annotations

import math

import numpy as np
import pytest

from entrain_infer import model


def test_the_clock_is_one_day() -> None:
    assert model.DAY_HOURS == 24.0
    assert model.OMEGA0 == pytest.approx(2.0 * math.pi / 24.0, rel=1e-15)


def test_minutes_and_rate_round_trip() -> None:
    """minutes_per_day and rad_per_hour are inverses, in both directions."""
    for minutes in (0.5, 2.0, 60.0, 1440.0):
        assert float(model.minutes_per_day(model.rad_per_hour(minutes))) == pytest.approx(
            minutes, rel=1e-12
        )


def test_one_radian_per_hour_is_a_day_of_slip_per_day() -> None:
    """b = OMEGA0 is one extra revolution per day, i.e. 1440 min of slip per day."""
    assert float(model.minutes_per_day(model.OMEGA0)) == pytest.approx(1440.0, rel=1e-12)


def test_minutes_from_phase_is_the_inverse_of_the_day() -> None:
    """A phase of 2 pi is a full day of timestamp offset."""
    assert float(model.minutes_from_phase(2.0 * math.pi)) == pytest.approx(1440.0, rel=1e-12)
    assert float(model.minutes_from_phase(0.0)) == 0.0


def test_conversions_vectorise() -> None:
    """The helpers are used on arrays of producers, not scalars."""
    b = np.array([0.0, 0.001, -0.001])
    minutes = model.minutes_per_day(b)
    assert minutes.shape == b.shape
    assert float(minutes[0]) == 0.0
    assert float(minutes[1]) > 0.0


def test_declared_constants_are_labelled_as_declared() -> None:
    """SIGMA_HONEST is 2 min/day of slip, and it is an assumption, not a measurement."""
    assert float(model.minutes_per_day(model.SIGMA_HONEST)) == pytest.approx(2.0, rel=1e-12)
    assert 0.0 < model.SIGMA_READ < 0.5
