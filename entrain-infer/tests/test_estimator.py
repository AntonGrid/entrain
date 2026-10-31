"""The two readings of a park: phase sees a constant shift, frequency sees a rate.

Every test here is against a series whose truth is known by construction -- a
sum of harmonics at ``OMEGA0`` with a controlled clock error -- so that an
estimator bug cannot hide behind weather.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from entrain_infer import estimator
from entrain_infer.model import OMEGA0, minutes_from_phase


def harmonic_series(samples: int, *, drift: float = 0.0, offset_hours: float = 0.0,
                    n_harm: int = 3, amplitude: float = 100.0, level: float = 1.0):
    """A rectified-day look-alike: harmonics of the park clock with a known error.

    ``drift`` is ``b`` (rad/h, positive = clock fast), ``offset_hours`` is ``tau``
    (hours, positive = clock ahead); both are applied to the *sampling grid*, so
    the series is exactly ``f(OMEGA0 * warped)`` and the readings must return
    ``+drift`` and ``+OMEGA0 * tau`` respectively.
    """
    t = np.arange(samples, dtype=float)
    warped = (1.0 + drift / OMEGA0) * t + offset_hours
    y = np.zeros(samples)
    for h in range(1, n_harm + 1):
        y += (amplitude / h) * np.cos(h * OMEGA0 * warped)
    return level * y


def test_phase_of_a_clean_cycle_is_zero() -> None:
    """A series on the park clock has no phase against the park clock."""
    y = harmonic_series(169)[None, :]
    assert float(estimator.estimate_phases(y)[0]) == pytest.approx(0.0, abs=0.05)


def test_phase_reads_a_constant_offset_in_minutes() -> None:
    """A device whose clock is ahead by tau reports a phase of +OMEGA0 * tau."""
    tau_hours = 0.5
    y = harmonic_series(169, offset_hours=tau_hours)[None, :]
    minutes = float(minutes_from_phase(estimator.estimate_phases(y)[0]))
    assert minutes == pytest.approx(60.0 * tau_hours, rel=0.02)


def test_frequency_is_blind_to_a_constant_offset() -> None:
    """A pure phase shift must not move the frequency: that is the gauge."""
    clean = harmonic_series(169)[None, :]
    shifted = harmonic_series(169, offset_hours=0.5)[None, :]
    assert float(estimator.estimate_offsets(shifted, 168)[0]) == pytest.approx(
        float(estimator.estimate_offsets(clean, 168)[0]), abs=1e-4
    )


@pytest.mark.parametrize("drift", [2e-3, -5e-3])
def test_frequency_recovers_a_known_rate(drift: float) -> None:
    """A clock running fast by b is measured as b."""
    y = harmonic_series(169, drift=drift)[None, :]
    measured, edge = estimator.estimate_offsets(y, 168, return_edge=True)
    assert not bool(edge[0]), "the estimate must not sit on the search band edge"
    assert float(measured[0]) == pytest.approx(drift, rel=0.05)


def test_frequency_refuses_a_window_shorter_than_a_revolution() -> None:
    """Inside one day there is no frequency to report -- refuse, do not guess."""
    y = harmonic_series(25)[None, :]
    with pytest.raises(ValueError, match="one revolution"):
        estimator.estimate_offsets(y, 24 - 1)


def test_phase_refuses_a_window_that_can_only_carry_one_bit() -> None:
    """Two samples are a sign, not an angle -- refuse, do not guess (MIN_PHASE_SAMPLES)."""
    y = harmonic_series(2, offset_hours=0.25)[None, :]
    with pytest.raises(ValueError, match="overdetermined"):
        estimator.estimate_phases(y)

    y4 = harmonic_series(4, offset_hours=0.25)[None, :]
    assert np.isfinite(float(estimator.estimate_phases(y4)[0]))


def test_both_readings_are_blind_to_level() -> None:
    """Scaling a producer's readings moves neither its phase nor its frequency."""
    base = harmonic_series(169)[None, :]
    loud = 1.6 * base
    assert float(estimator.estimate_offsets(loud, 168)[0]) == pytest.approx(
        float(estimator.estimate_offsets(base, 168)[0]), abs=1e-6
    )
    assert float(estimator.estimate_phases(loud)[0]) == pytest.approx(
        float(estimator.estimate_phases(base)[0]), abs=1e-9
    )


def test_offsets_are_read_row_by_row() -> None:
    """The reading is per producer: a clock error on one row leaves the others alone."""
    honest = harmonic_series(169)
    series = np.vstack([honest, harmonic_series(169, drift=5e-3)])
    offsets, edges = estimator.estimate_offsets(series, 168, return_edge=True)
    assert not edges.any()
    assert abs(float(offsets[0])) < 1e-3
    assert float(offsets[1]) == pytest.approx(5e-3, rel=0.05)


def test_demodulation_removes_the_park_clock() -> None:
    """``phasor`` of a pure cycle at OMEGA0 is real and positive."""
    y = np.cos(OMEGA0 * np.arange(49.0))[None, :]
    value = estimator.phasor(y)[0]
    assert value.real > 0
    assert abs(value.imag) < 1e-9 * max(abs(value.real), 1.0) + 1e-6
