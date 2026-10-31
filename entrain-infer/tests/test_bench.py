"""The bench keeps its promises: real background, declared injections, honest map.

The important test here is ``test_constant_offset_is_a_gauge_shift``: it pins the
M1 claim that the same lie is loud in one reading and silent in the other, on a
series whose truth is known exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from entrain_infer import bench, detector, estimator
from entrain_infer.model import OMEGA0, minutes_from_phase, minutes_per_day, rad_per_hour

DATA = bench.DATA_DIR / "nasa_power_2021_vladimir.json"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="background not fetched")


def harmonic_series(
    samples: int,
    *,
    drift: float = 0.0,
    offset_hours: float = 0.0,
    amplitude: float = 100.0,
) -> np.ndarray:
    """A rectified-day look-alike: three harmonics of the park clock, clock error known."""
    t = np.arange(samples, dtype=float)
    warped = t * (OMEGA0 / (OMEGA0 + drift)) + offset_hours
    y = np.zeros(samples)
    for h in range(1, 4):
        y += (amplitude / h) * np.cos(h * OMEGA0 * warped)
    return y


def honest_park(n: int = 12, samples: int = 169) -> np.ndarray:
    return np.vstack([harmonic_series(samples) for _ in range(n)])


# ─────────────────────────────────────────────────────────────── background ──


@needs_data
def test_background_is_real_and_shaped() -> None:
    """The fetched background is a year of hourly irradiance, not a placeholder."""
    bg = bench.load_background("nasa_power_2021")
    assert bg.size == 8760
    assert bg.max() > 500.0
    assert bg.min() >= 0.0
    assert float((bg == 0).mean()) > 0.3, "night must exist"


@needs_data
def test_second_source_agrees_in_scale() -> None:
    """Two independent datasets describe the same sky."""
    a = bench.load_background("nasa_power_2021")
    b = bench.load_background("pvgis_2020")
    assert abs(float(a.mean()) - float(b.mean())) < 0.5 * float(a.mean())


@needs_data
def test_park_is_deterministic_and_seeded() -> None:
    """Same seed, same park; different seed, different park."""
    bg = bench.load_background("nasa_power_2021")
    p1 = bench.build_park(bg, 8, 24, "july", seed=3)
    p2 = bench.build_park(bg, 8, 24, "july", seed=3)
    p3 = bench.build_park(bg, 8, 24, "july", seed=4)
    assert p1.series.shape == (8, 25)
    assert np.array_equal(p1.series, p2.series)
    assert not np.array_equal(p1.series, p3.series)


# ─────────────────────────────────────────────────────────────── injections ──


def test_level_injection_touches_one_row_and_only_scale() -> None:
    """The level lie is multiplicative and local to the guilty row."""
    series = honest_park(n=4)
    out = bench.inject_level(series, [2], 1.6)
    assert np.allclose(out[2], 1.6 * series[2])
    for i in (0, 1, 3):
        assert np.array_equal(out[i], series[i])


def test_step_injection_changes_the_row_only_after_t0() -> None:
    """A clock that jumps at t0 leaves everything before t0 untouched."""
    series = honest_park(n=2)
    out = bench.inject_timestamp_step(series, [1], 2.0, 72)
    assert np.array_equal(out[0], series[0])
    assert np.allclose(out[1, :72], series[1, :72])
    assert not np.allclose(out[1, 72:], series[1, 72:])


def test_clock_rate_injection_is_read_back_as_b() -> None:
    """The rate lie is the H1 of the test, so the reading must return it."""
    series = honest_park(n=2)
    out = bench.inject_clock_rate(series, [1], 4e-3)
    measured = estimator.estimate_offsets(out, 168)
    assert float(measured[1]) == pytest.approx(4e-3, rel=0.06)
    assert abs(float(measured[0])) < 1e-3


# ─────────────────────────────────────────────────────────────── the gauge ──


def test_constant_offset_is_a_gauge_shift() -> None:
    """A constant timestamp offset is loud in phase and silent in frequency."""
    tau_hours = 0.5
    series = np.vstack(
        [harmonic_series(169, offset_hours=tau_hours)]
        + [harmonic_series(169) for _ in range(9)]
    )

    by_phase = detector.detect(estimator.estimate_phases(series), 0.02, 1e-3)
    assert by_phase.flagged and by_phase.node == 0
    assert float(minutes_from_phase(by_phase.b_hat)) == pytest.approx(
        60.0 * tau_hours, rel=0.02
    )

    by_offset = detector.detect(estimator.estimate_offsets(series, 168), 1e-4, 1e-3)
    assert not by_offset.flagged, "a pure phase shift must not move the frequency"


def test_step_is_loud_in_phase_and_partial_in_frequency() -> None:
    """A jump is a phase lie, so the two readings split it -- and both see it.

    Half of a weekly window carries a 2 h jump, so the phase reading is about
    1 h; the frequency reading is not a shift at all but a corrupted harmonic
    fit, and it lands at ~40 % of that. Both magnitudes are measured, not
    derived: the point of the test is that neither reading is blind to a step,
    unlike a constant offset which the frequency cannot see at all.
    """
    series = honest_park(n=6)
    liars = bench.inject_timestamp_step(series, [0], 2.0, 84)
    honest_phase = float(np.angle(np.mean(np.exp(1j * estimator.estimate_phases(series)[1:]))))

    by_phase = float(minutes_from_phase(estimator.estimate_phases(liars)[0] - honest_phase))
    assert by_phase == pytest.approx(60.0, abs=10.0)

    by_offset = float(minutes_per_day(estimator.estimate_offsets(liars, 168)[0]))
    assert 10.0 < by_offset < 40.0
    assert detector.detect(estimator.estimate_phases(liars), 0.02, 1e-3).flagged


def test_collusion_dilutes_the_statistic_not_the_reading() -> None:
    """k identical liars read as (n-k)/(n-1) of one liar in t, though each reads b.

    The reading of a liar is a per-producer number and does not know about its
    accomplices; the *evidence* does, because the profiled mean moves with them.
    That is exactly why collusion is cheap, and why M2 has to bound it.
    """
    n, k, tau = 30, 3, 2.0
    liars = bench.inject_timestamp_step(honest_park(n=n), list(range(k)), tau, 84)
    single = bench.inject_timestamp_step(honest_park(n=n), [0], tau, 84)

    x = estimator.estimate_offsets(liars, 168)
    t_group = float(detector.studentize(x)[0])
    t_single = float(detector.studentize(estimator.estimate_offsets(single, 168))[0])
    assert t_single / t_group == pytest.approx((n - 1) / (n - k), rel=1e-6)
    assert detector.collusion_dilution(n, k) == pytest.approx((n - k) / (n - 1), rel=1e-12)

    # b_hat never shrinks: the statistic does. A liar's own reading is undiluted.
    verdict = detector.detect(x, 1e-4, 1e-3)
    assert verdict.b_hat == pytest.approx(float(x[0] - np.delete(x, 0).mean()), rel=1e-9)


# ─────────────────────────────────────────────────────── the reading frame ──


def test_centre_phases_is_a_gauge_rotation() -> None:
    """Rotating a park cannot change what the detector says about it.

    The statistic is built from node-to-node differences, so a common angle is
    invisible to it. ``centre_phases`` exists only to keep the linear arithmetic
    away from the +-pi seam: a park whose common phase sits near pi would have its
    honest nodes split across the seam, and the detector would flag arithmetic
    instead of a liar.
    """
    rng = np.random.default_rng(7)
    phases = rng.normal(0.0, 0.005, 8)  # honest park: tight, but not degenerate
    straddling = np.angle(np.exp(1j * (phases + math.pi)))
    assert float(np.ptp(straddling)) > 6.0, "the trap must actually be a trap"

    raw = detector.detect(straddling, 0.02, 1e-3)
    assert raw.flagged and raw.statistic > 100.0, "uncentred, arithmetic flags the park"

    centred = detector.detect(estimator.centre_phases(straddling), 0.02, 1e-3)
    assert not centred.flagged
    assert centred.statistic == pytest.approx(
        detector.detect(phases, 0.02, 1e-3).statistic, rel=1e-9
    )


def test_two_samples_are_refused_not_dressed_up_as_a_phase() -> None:
    """One bit is not an angle, and the estimator says so instead of printing one.

    With two samples the DC-profiled phasor is ``(y1 - y0)(e^{-i w0} - 1)``: a
    positive scalar times a fixed complex number, so only the *sign* of the change
    survives -- exactly two possible readings, a half turn apart. The measured σ of
    such a pool describes the day/night mix of the window, not the instrument,
    which is why ``estimate_phases`` refuses it the way ``estimate_offsets``
    refuses a window shorter than a revolution.
    """
    rising = np.array([0.3, 1.0])
    falling = np.array([1.0, 0.3])
    two = estimator.phasor(np.vstack([rising, falling]))
    assert np.angle(two[0]) == pytest.approx(-math.pi / 2 - OMEGA0 / 2, abs=1e-12)
    assert np.angle(two[1]) - np.angle(two[0]) == pytest.approx(math.pi, abs=1e-9)

    with pytest.raises(ValueError, match="one bit|sign of the change"):
        estimator.estimate_phases(np.vstack([rising, falling]))

    # four samples are the first window that fits instead of interpolating
    y = np.vstack([np.cos(OMEGA0 * np.arange(4) + 0.7)] * 2)
    assert float(estimator.estimate_phases(y)[0]) == pytest.approx(0.7, abs=0.2)


@needs_data
def test_probe_separates_the_instrument_gap_from_the_effect_seen() -> None:
    """``read_gap`` is the physical lie; ``b_eff`` is what the statistic sees.

    Even a single liar reads as ``(n-1)/n`` of its gap, because the mean it is
    compared against contains the liar itself; a group of ``k`` adds the
    ``(n-k)/(n-1)`` of collusion on top. Measuring both is what makes §9 a
    measurement rather than a restatement of the formula.
    """
    bg = bench.load_background("nasa_power_2021")
    sigma = bench.measure_sigma(bg, 8, 24, "july", observable="phase", repeats=4)["sigma"]
    n = 8

    def run(k: int) -> dict:
        nodes = list(range(k))
        return bench.probe(
            bg, n, 24, "july", sigma,
            lambda series, r, _k=k: (bench.inject_timestamp_step(series, nodes, 2.0, 0), nodes),
            observable="phase", alpha=1e-3, repeats=3,
        )

    single, pair = run(1), run(2)
    assert single["read_gap_minutes"] == pytest.approx(120.0, rel=0.05)
    assert pair["read_gap_minutes"] == pytest.approx(120.0, rel=0.05)
    # the profiled mean costs (n-1)/n even with no colluder in the group
    assert single["b_eff"] == pytest.approx(
        single["read_gap"] * (n - 1) / n, rel=0.02
    )
    # and the group is diluted further, exactly by the collusion factor
    assert pair["b_eff"] / single["b_eff"] == pytest.approx((n - 2) / (n - 1), rel=0.05)


# ──────────────────────────────────────────────────────────────────── noise ──


@needs_data
def test_measured_sigma_shrinks_with_the_window() -> None:
    """A week reads the park's frequency better than a day."""
    bg = bench.load_background("nasa_power_2021")
    day = bench.measure_sigma(bg, 10, 24, "july", repeats=12)
    week = bench.measure_sigma(bg, 10, 168, "july", repeats=12)
    assert 0.0 < week["sigma"] < day["sigma"]


@needs_data
def test_measured_sigma_is_finite_and_reported_in_minutes() -> None:
    """sigma has units, and the minutes conversion is the declared one."""
    bg = bench.load_background("nasa_power_2021")
    week = bench.measure_sigma(bg, 10, 168, "july", repeats=8)
    assert math.isfinite(week["sigma_minutes"])
    assert week["unit"] == "rad/h"
    assert week["sigma_minutes"] == pytest.approx(float(
        week["sigma"] * 1440.0 * 24.0 / (2.0 * math.pi)
    ), rel=1e-12)


def test_circular_sigma_is_scale_free() -> None:
    """A uniform phase has a large circular sigma; a tight one has a small one."""
    rng = np.random.default_rng(5)
    assert bench.circular_sigma(rng.uniform(-math.pi, math.pi, 4000)) > 1.5
    assert bench.circular_sigma(rng.normal(0.0, 0.01, 4000)) == pytest.approx(0.01, rel=0.05)


def test_read_dispatches_on_the_observable() -> None:
    """One entry point, two readings, and a loud failure for a third."""
    series = honest_park(n=3)
    assert bench.read(series, 168, "offset").shape == (3,)
    assert bench.read(series, 168, "phase").shape == (3,)
    with pytest.raises(ValueError, match="unknown observable"):
        bench.read(series, 168, "amplitude")


def test_unmeasurable_cells_are_declared_not_discovered() -> None:
    """The grid refuses the cells the data cannot fill, and says why in words.

    Both refusals are declared constants (``MIN_READ_HOURS``, ``MIN_PHASE_SAMPLES``),
    so the grid must skip exactly the cells below them and nothing else: an hourly
    window has no revolution to measure a frequency against, and two samples are a
    sign rather than an angle. Printing a σ there would measure the declaration.
    """
    assert bench.unmeasurable(1, "offset") is not None
    assert bench.unmeasurable(24, "offset") is None
    assert bench.unmeasurable(168, "offset") is None
    assert bench.unmeasurable(1, "phase") is not None
    assert bench.unmeasurable(24, "phase") is None
    assert "оборот" in str(bench.unmeasurable(1, "offset"))
    assert "знак изменения" in str(bench.unmeasurable(1, "phase"))


def test_criteria_are_preregistered() -> None:
    """The thresholds of the run are in a file, not in the code that measures them."""
    criteria = bench.load_criteria()
    assert 0.0 < float(criteria["alpha"]) < 0.05
    assert float(criteria["target_power"]) == pytest.approx(0.9)
    assert set(criteria["ns"]) == {10, 30, 100}
    assert set(criteria["hours"]) == {1, 24, 168}
    assert len(criteria["acceptance"]) == 8
    assert criteria["declared_constants"]["monte_carlo_seed"] == 20261030
    assert criteria["declared_before_run"] is True



