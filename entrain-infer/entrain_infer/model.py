"""Declared model of the detector: units, constants, one observation.

Everything a claim in `bench/report/REPORT.md` depends on is here, so that a
number can be traced to the assumption that produced it.

The park's clock is the day: one revolution per 24 h, ``OMEGA0 = 2*pi/24``
rad/h. A producer's state is read against that clock in exactly two ways (see
``estimator``): its *frequency* offset (rad/h, a sustained rate lie) and its
*phase* (rad, a constant offset lie). Both feed the same GLRT; nothing else is
observed -- no level, no amplitude.

This module also carries the two constants that are **declared, not derived**,
and says so where they live:

* ``SIGMA_HONEST`` -- how far honest producers' own cycles drift apart, in the
  same unit as ``b``. 2 minutes of cycle slip per day. It is an assumption
  about a park, and every table in the report is labelled with it;
* ``SIGMA_READ`` -- relative reading noise of one hourly bucket (2 % of the
  daily amplitude), used only to build a semi-synthetic park.

Neither is a threshold: thresholds come from ``detector.null_threshold`` and
from the operator's risk ``alpha``, which is the point of M1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The park's clock: one revolution per day, radians per hour.
DAY_HOURS: float = 24.0
OMEGA0: float = 2.0 * math.pi / DAY_HOURS  # 0.261799 rad/h

#: One rad/hour of frequency offset, written as cycle slip per day, in minutes:
#: ``24 h * b rad`` of accumulated phase is ``24 b / 2 pi`` revolutions, so
#: ``minutes_per_day(OMEGA0) = 1440``: a whole extra revolution per day.
MINUTES_PER_DAY_PER_RAD_PER_HOUR: float = 60.0 * 24.0 * 24.0 / (2.0 * math.pi)


def minutes_per_day(b: float | np.ndarray) -> float | np.ndarray:
    """Convert a frequency offset (rad/h) into minutes of cycle slip per day."""
    return np.asarray(b, dtype=float) * MINUTES_PER_DAY_PER_RAD_PER_HOUR


def rad_per_hour(minutes: float) -> float:
    """Convert minutes of cycle slip per day into a frequency offset (rad/h)."""
    return float(minutes) / MINUTES_PER_DAY_PER_RAD_PER_HOUR


def minutes_from_phase(phi: float | np.ndarray) -> float | np.ndarray:
    """Convert a phase offset (rad) into minutes of timestamp offset."""
    return np.asarray(phi, dtype=float) * (DAY_HOURS * 60.0) / (2.0 * math.pi)


#: Declared spread of honest producers' own frequencies: 2 min of slip per day.
SIGMA_HONEST: float = rad_per_hour(2.0)

#: Declared relative reading noise of one hourly bucket (semi-synthetic park).
SIGMA_READ: float = 0.02


@dataclass(frozen=True)
class Observation:
    """One reading of a park: per-producer frequency offsets and their noise.

    ``x[i]`` is producer ``i``'s frequency offset against the park clock, in
    rad/h, averaged over ``hours``. ``sigma`` is the standard deviation of one
    such number under H0 -- honest spread plus estimation noise, measured on the
    background and reported next to every table.
    """

    x: np.ndarray
    sigma: float
    hours: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.x.ndim != 1:
            raise ValueError("x must be one-dimensional (one number per producer)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.hours <= 0:
            raise ValueError("hours must be positive")

    @property
    def n(self) -> int:
        return int(self.x.size)

    def standardized(self) -> np.ndarray:
        """``x / sigma`` -- the units the GLRT works in."""
        return self.x / self.sigma

    def minutes_per_day(self) -> np.ndarray:
        """The same offsets as cycle slip per day, in minutes (human units)."""
        return minutes_per_day(self.x)

    def b_minutes_per_day(self) -> float:
        """The detector's resolution in minutes of slip per day."""
        return float(minutes_per_day(self.sigma))
