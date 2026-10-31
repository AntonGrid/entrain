"""Reading a park: two observables, both taken against the park clock.

The park clock is the day: one revolution per 24 h. From a producer's own hourly
series a narrowband reading can extract exactly two numbers, and which one you
use decides which lie you can see:

``phase``  the fundamental of a joint harmonic fit at ``OMEGA0`` (DC and the
           higher harmonics profiled out). Units: radians. Sees a **constant**
           offset of a device's timestamps in full, and half of a **step** --
           half because only the samples after the jump carry it.
``offset`` the frequency of least harmonic-regression residual inside a declared
           band around ``OMEGA0``, refined by three-point parabolic interpolation.
           Units: rad/h. Sees a **sustained rate** lie (a clock that runs fast)
           and a *fraction* of a **step**, which corrupts the harmonic fit
           without being a rate at all (measured: about 40 % of the step as
           phase). A constant offset is a pure phase shift and does not move the
           frequency at all -- that is the gauge symmetry M1 rests on.

Both come from the same demodulation, both go into the same GLRT, and neither
presupposes a threshold.

Two limits are declared, not discovered:

* **A frequency needs a revolution.** ``MIN_READ_HOURS`` is 24: inside one day
  the daily cycle has not closed, so its frequency is not identifiable and
  ``estimate_offsets`` refuses instead of returning a confident lie.
* **A phase needs an overdetermined fit.** ``MIN_PHASE_SAMPLES`` is 4: DC plus
  one harmonic is already three parameters, so fewer than four samples determine
  nothing. On two samples the reading is not a phase at all -- it is one bit,
  whether the series rose or fell (the phasor is a positive scalar times a fixed
  complex number, so only its sign survives). ``estimate_phases`` refuses too.
* **The search band is a prior.** ``SEARCH_BAND`` is +-0.05 rad/h (+-275 minutes
  of slip per day); outside it the park is not "honest producers plus one liar",
  it is a different mechanism. Estimates that land on the edge are flagged.

Sun-driven series have no phase at night; zeros simply carry no weight in a
periodogram, so no masking is needed and none is done.

Nothing here decides anything; ``detector`` does that.
"""
from __future__ import annotations

import numpy as np

from .model import DAY_HOURS, OMEGA0

#: A frequency is not identifiable inside one revolution (declared).
MIN_READ_HOURS: float = DAY_HOURS

#: Samples a phase needs: DC plus one harmonic is three parameters, so four
#: samples are the first window that fits rather than interpolates (declared).
MIN_PHASE_SAMPLES: int = 4

#: Declared band of the periodogram search, rad/h (+-275 min of slip per day).
SEARCH_BAND: float = 0.05

#: Grid points across the band; the parabolic refinement, not the grid, sets the
#: resolution.
GRID: int = 401

#: Harmonics in the frequency regression. Three covers the day/night edge and the
#: seasonal shoulder of a mid-latitude irradiance profile; more only fit clouds.
N_HARM: int = 3


def demodulate(series: np.ndarray, omega0: float = OMEGA0) -> np.ndarray:
    """``(y - mean) * exp(-i omega0 t)`` -- the series seen from the park clock."""
    y = np.asarray(series, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    t = np.arange(y.shape[1], dtype=float)
    centred = y - y.mean(axis=1, keepdims=True)
    return centred * np.exp(-1j * omega0 * t)[None, :]


def phasor(series: np.ndarray, omega0: float = OMEGA0) -> np.ndarray:
    """``sum_t u(t)``: each producer's phasor against the park clock, complex."""
    return demodulate(series, omega0).sum(axis=1)


def _phase_basis(samples: int, n_harm: int) -> np.ndarray:
    """``[1, cos(w0 t), sin(w0 t), cos(2 w0 t), ...]`` -- DC first, then pairs."""
    t = np.arange(samples, dtype=float)
    cols = [np.ones(samples)]
    for h in range(1, n_harm + 1):
        cols.append(np.cos(h * OMEGA0 * t))
        cols.append(np.sin(h * OMEGA0 * t))
    return np.stack(cols, axis=1)


def estimate_phases(series: np.ndarray, *, n_harm: int = N_HARM) -> np.ndarray:
    """Phase of every producer against the park clock, radians.

    A device whose clock is ahead by ``tau`` reads the future, so its series is
    ``f(OMEGA0 (t + tau))`` and the fundamental's complex amplitude carries this
    phase: ``+OMEGA0 * tau``, which ``minutes_from_phase`` prints as ``+60 tau``
    minutes. No other reading sees a pure timestamp offset.

    Taking the phase from the *fundamental of a joint harmonic fit* rather than
    from a bare zero-lag DFT removes the leakage that the rest of the day's shape
    leaks into the ``OMEGA0`` bin whenever the window is not an exact number of
    days (a week plus a sample leaks about 2 % of the offset into the reading).
    That leakage is a calibration bias, not noise, and it is worth removing in an
    observable whose whole job is minutes of offset. The number of harmonics is
    capped at half the samples so that the fit stays overdetermined, and a window
    with fewer than ``MIN_PHASE_SAMPLES`` samples is refused: on two samples the
    phasor is a positive scalar times a fixed complex number, so the reading is
    the sign of the change and nothing else -- one bit wearing the name of an
    angle. ``estimate_offsets`` refuses short windows for the same reason.
    """
    y = np.asarray(series, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    samples = y.shape[1]
    if samples < MIN_PHASE_SAMPLES:
        raise ValueError(
            f"a phase needs an overdetermined fit: {samples} samples < "
            f"{MIN_PHASE_SAMPLES} (two samples carry the sign of the change, "
            "not a phase)"
        )
    fitted = min(n_harm, (samples - 2) // 2)
    basis = _phase_basis(samples, fitted)
    coef, *_ = np.linalg.lstsq(basis, y.T, rcond=None)  # (1 + 2 * fitted, rows)
    return np.angle(coef[1] - 1j * coef[2])


def centre_phases(phases: np.ndarray) -> np.ndarray:
    """Rotate a park's phases to their own circular mean, then unwrap into (-pi, pi].

    The GLRT statistic is built from *differences* between nodes, so adding the
    same angle to every node leaves every ``t_k`` unchanged: this rotation is a
    pure choice of gauge and cannot change a verdict. What it buys is arithmetic
    safety -- the detector uses linear residuals, and a park whose common phase
    happens to sit near ``+-pi`` would otherwise have its honest nodes split
    across the seam and be flagged for it. The park's common phase is not an
    observable anyone can use (every producer could share it); only the
    deviation is.
    """
    p = np.asarray(phases, dtype=float).ravel()
    if p.size == 0:
        return p
    return np.angle(np.exp(1j * (p - np.angle(np.mean(np.exp(1j * p))))))


def _harmonic_design(omegas: np.ndarray, samples: int, n_harm: int) -> np.ndarray:
    """``cos(h w t), sin(h w t)`` for every candidate ``w``: shape (G, 2H, M)."""
    t = np.arange(samples, dtype=float)
    blocks = []
    for h in range(1, n_harm + 1):
        blocks.append(np.cos(h * np.outer(omegas, t)))
        blocks.append(np.sin(h * np.outer(omegas, t)))
    return np.stack(blocks, axis=1)


def estimate_offsets(
    series: np.ndarray,
    hours: int | None = None,
    *,
    band: float = SEARCH_BAND,
    grid: int = GRID,
    n_harm: int = N_HARM,
    return_edge: bool = False,
):
    """Frequency offset of every producer against the park clock, rad/h.

    A rectified day/night profile is **not** narrowband: its harmonics are of the
    same order as its fundamental, so a plain periodogram peak is dragged around
    by the shape of the day. The estimator therefore fits a harmonic regression
    (``DC`` profiled out, ``N_HARM`` harmonics) at every candidate frequency and
    takes the frequency of least residual; the harmonics are nuisance parameters,
    the frequency is the signal. Three-point parabolic refinement in ``log RSS``
    then sets the resolution, not the grid.

    ``series`` is ``(n_producers, samples)`` hourly values; ``hours`` keeps the
    trailing ``hours + 1`` samples so the window spans exactly ``hours`` hours.
    Raises if the window is shorter than one revolution -- a daily cycle observed
    for less than a day has no frequency to report, whatever an estimator says.
    """
    y = np.asarray(series, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    span = float(hours if hours is not None else y.shape[1] - 1)
    if span < MIN_READ_HOURS:
        raise ValueError(
            f"a frequency needs one revolution: {span:g} h < {MIN_READ_HOURS:g} h "
            "(use estimate_phases for windows shorter than a day)"
        )
    if hours is not None:
        y = y[:, -(int(hours) + 1) :]
    y = y - y.mean(axis=1, keepdims=True)

    omegas = OMEGA0 + np.linspace(-band, band, grid)
    design = _harmonic_design(omegas, y.shape[1], n_harm)  # (G, 2H, M)
    n_freq, n_cols = design.shape[0], design.shape[1]

    gram = np.einsum("gat,gbt->gab", design, design)
    rhs = np.einsum("gat,it->gai", design, y)
    coef = np.linalg.solve(gram + 1e-12 * np.eye(n_cols), rhs)
    explained = np.einsum("gai,gai->gi", coef, rhs)
    rss = np.maximum((y * y).sum(axis=1)[None, :] - explained, 1e-30)

    k = np.argmin(rss, axis=0)
    rows = np.arange(y.shape[0])
    edges = (k == 0) | (k == n_freq - 1)
    idx = np.clip(k, 1, n_freq - 2)

    a = np.log(rss[idx - 1, rows])
    b = np.log(rss[idx, rows])
    c = np.log(rss[idx + 1, rows])
    denom = a - 2.0 * b + c
    denom = np.where(np.abs(denom) > 0, denom, 1.0)
    shift = np.clip(0.5 * (a - c) / denom, -0.5, 0.5)
    step = float(omegas[1] - omegas[0])
    offsets = omegas[idx] + shift * step - OMEGA0
    offsets = np.where(edges, omegas[k] - OMEGA0, offsets)

    if return_edge:
        return offsets, edges
    return offsets



def read_park(series: np.ndarray, hours: int | None = None, source: str = ""):
    """``estimate_offsets`` wrapped into a ``model.Observation`` (sigma left 1)."""
    from .model import Observation

    x = estimate_offsets(series, hours)
    span = float(hours if hours is not None else series.shape[-1] - 1)
    return Observation(x=x, sigma=1.0, hours=span, source=source)

