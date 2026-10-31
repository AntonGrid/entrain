"""Semi-synthetic bench: real background, declared injections, measured noise.

The background is **real**: hourly all-sky shortwave irradiance for the Vladimir
region (56.13 N, 40.40 E), two independent sources -- NASA POWER 2021 and PVGIS
2020. A park of ``n`` producers is built from it: every producer is the same
weather envelope with its own capacity, its own reading noise and its own honest
frequency drift (``SIGMA_HONEST``). That gives a background with real clouds,
real seasons and real night gaps, and an *exactly known* ground truth for the
injections -- the only way to have both at once.

Two observables are measured on every park (``estimator``): the frequency offset
against the park clock (rad/h) and the phase (rad). The same GLRT is applied to
both, so the bench can say *which* lie each reading can see -- and the map is not
the identity:

``timestamp_step``
    the device's clock jumps by ``tau`` hours at ``t0`` and stays there.
``constant_offset``
    the clock is off by ``tau`` hours for the whole window -- a pure phase shift.
    Visible in the phase, invisible in the frequency. A negative control.
``clock_rate``
    the clock runs fast by ``b`` rad/h for the whole window: the H1 alternative
    the frequency GLRT is built for.
``level``
    the device multiplies its readings. Physics is blind to it by construction.
``collusion``
    ``k`` devices take the same step; the surviving lie should shrink as
    ``(n-k)/(n-1)``.

Nothing here decides a threshold: ``alpha`` comes from ``bench/criteria.json``,
written before the run.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from . import detector, estimator
from .model import (
    OMEGA0,
    SIGMA_HONEST,
    SIGMA_READ,
    minutes_from_phase,
    minutes_per_day,
)

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent / "bench"
DATA_DIR = BENCH_DIR / "data"

#: Declared windows: the 15th of four months at 00:00, so that a season is a
#: named choice and not a lucky slice.
SEASONS: dict[str, int] = {
    "january": 14 * 24,
    "april": 104 * 24,
    "july": 195 * 24,
    "october": 287 * 24,
}

#: The two readings the GLRT can run on.
OBSERVABLES: tuple[str, ...] = ("offset", "phase")

#: Declared producer counts and window lengths of the M1 grid.
NS: tuple[int, ...] = (10, 30, 100)
HOURS: tuple[int, ...] = (1, 24, 168)


# ──────────────────────────────────────────────────────────────── background ──


def load_nasa_power(path: Path | None = None) -> np.ndarray:
    """Hourly ``ALLSKY_SFC_SW_DWN`` (W/m^2) from the NASA POWER JSON."""
    path = path or DATA_DIR / "nasa_power_2021_vladimir.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    series = raw["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    keys = sorted(series)
    return np.array([float(series[k]) for k in keys], dtype=float)


def load_pvgis(path: Path | None = None) -> np.ndarray:
    """Hourly global irradiance ``G(i)`` (W/m^2) from the PVGIS JSON."""
    path = path or DATA_DIR / "pvgis_2020_vladimir.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return np.array([float(h["G(i)"]) for h in raw["outputs"]["hourly"]], dtype=float)


def load_background(name: str = "nasa_power_2021") -> np.ndarray:
    """One of the two real backgrounds, by declared name."""
    if name == "nasa_power_2021":
        return load_nasa_power()
    if name == "pvgis_2020":
        return load_pvgis()
    raise ValueError(f"unknown background {name!r}")


@dataclass(frozen=True)
class Park:
    """A semi-synthetic park and what it was built from."""

    series: np.ndarray  # (n_producers, samples)
    background: str
    season: str
    hours: int
    start: int

    @property
    def n(self) -> int:
        return int(self.series.shape[0])

    @property
    def samples(self) -> int:
        return int(self.series.shape[1])


def build_park(
    background: np.ndarray,
    n: int,
    hours: int,
    season: str = "july",
    *,
    source: str = "nasa_power_2021",
    sigma_honest: float = SIGMA_HONEST,
    sigma_read: float = SIGMA_READ,
    seed: int = 1,
) -> Park:
    """Build ``n`` producers out of one real window of the background."""
    start = SEASONS[season]
    need = hours + 1
    if start + need > background.size:
        raise ValueError(f"background too short for {need} samples from {start}")
    window = background[start : start + need]
    t = np.arange(need, dtype=float)

    rng = np.random.default_rng(seed)
    capacity = rng.lognormal(mean=0.0, sigma=0.10, size=n)
    offset = rng.uniform(0.02, 0.20, size=n) * max(window.mean(), 1.0)
    drift = rng.normal(0.0, sigma_honest, size=n)
    noise = sigma_read * max(float(np.abs(window).max()), 1.0)

    series = np.empty((n, need), dtype=float)
    for i in range(n):
        # a clock running at omega0 + drift reads the weather earlier by drift
        s = t * (OMEGA0 / (OMEGA0 + drift[i]))
        series[i] = capacity[i] * np.interp(s, t, window) + offset[i]
    series += rng.normal(0.0, noise, size=series.shape)
    return Park(series=series, background=source, season=season, hours=hours, start=start)


# ─────────────────────────────────────────────────────────────── injections ──


def inject_timestamp_step(
    series: np.ndarray, nodes: list[int], tau_hours: float, t0: int
) -> np.ndarray:
    """The clock jumps forward by ``tau`` hours at ``t0``: the row is the truth shifted.

    ``tau > 0`` means the device's timestamps jumped *ahead*, so after ``t0`` its
    row shows what the sky did ``tau`` hours later. The last samples of a shifted
    row have no future to read and are held at the edge value; with a step in the
    middle of a window (the only place a step is measurable) that is a handful of
    samples out of a week.
    """
    out = np.asarray(series, dtype=float).copy()
    m = out.shape[1]
    t = np.arange(m, dtype=float)
    for i in nodes:
        shifted = np.interp(t + tau_hours, t, out[i])
        out[i, t0:] = shifted[t0:]
    return out


def inject_clock_rate(series: np.ndarray, nodes: list[int], b: float) -> np.ndarray:
    """The clock runs fast by ``b`` rad/h for the whole window (the H1 lie).

    A device whose clock is ahead reads the future, so its row is the truth at
    ``t * (1 + b/OMEGA0)``; a harmonic series then has frequency ``OMEGA0 + b`` and
    the reading returns ``+b``. The warp is centred on the window so that the
    unavoidable truncation is half of the total drift at each end, not all of it
    at one end; the few clamped samples are the price of a finite background and
    are mentioned in the report rather than hidden.
    """
    out = np.asarray(series, dtype=float).copy()
    m = out.shape[1]
    t = np.arange(m, dtype=float)
    centre = 0.5 * (m - 1)
    s = centre + (1.0 + b / OMEGA0) * (t - centre)
    for i in nodes:
        out[i] = np.interp(s, t, out[i])
    return out


def inject_level(series: np.ndarray, nodes: list[int], factor: float) -> np.ndarray:
    """The device multiplies its readings: level, not phase, not frequency."""
    out = np.asarray(series, dtype=float).copy()
    for i in nodes:
        out[i] *= factor
    return out


# ───────────────────────────────────────────────────────────────── readings ──


def read(series: np.ndarray, hours: int, observable: str = "offset") -> np.ndarray:
    """One reading of the park: frequency offsets (rad/h) or phases (rad).

    For the phase the park is rotated to its own circular mean before it leaves
    this function (``estimator.centre_phases``): the rotation is a gauge choice
    that leaves every ``t_k`` unchanged, and it keeps the detector's linear
    arithmetic away from the ``+-pi`` seam. For the offset the readings are
    already small numbers around zero, and the detector profiles the common mean
    out, so nothing needs centring.
    """
    if observable == "offset":
        return estimator.estimate_offsets(series, hours)
    if observable == "phase":
        window = np.asarray(series, dtype=float)[:, -(int(hours) + 1) :]
        return estimator.centre_phases(estimator.estimate_phases(window))
    raise ValueError(f"unknown observable {observable!r}")


def circular_sigma(phases: np.ndarray) -> float:
    """Circular standard deviation: ``sqrt(-2 ln R)`` with ``R`` the mean resultant."""
    if phases.size == 0:
        return float("nan")
    resultant = abs(np.mean(np.exp(1j * phases)))
    return float(np.sqrt(max(-2.0 * math.log(max(resultant, 1e-12)), 0.0)))


def measure_sigma(
    background: np.ndarray,
    n: int,
    hours: int,
    season: str,
    *,
    observable: str = "offset",
    source: str = "nasa_power_2021",
    repeats: int = 200,
    seed: int = 7,
) -> dict[str, object]:
    """Measure the null scatter of one reading -- honest spread plus reading noise.

    Each repeat is a fresh honest park. Readings are centred inside each park
    before pooling, because the GLRT profiles the common mean out: what matters is
    scatter around the park, not the park's own idea of the clock.
    """
    pooled = []
    for r in range(repeats):
        park = build_park(background, n, hours, season, source=source, seed=seed + r)
        x = read(park.series, hours, observable)
        if observable == "phase":
            x = estimator.centre_phases(x)  # idempotent, kept for the invariant
        else:
            x = x - x.mean()
        pooled.append(x)
    pooled = np.concatenate(pooled)
    if observable == "phase":
        sigma = circular_sigma(pooled)
        minutes = float(minutes_from_phase(sigma))
        excess = float("nan")
    else:
        sigma = float(np.std(pooled, ddof=1))
        minutes = float(minutes_per_day(sigma))
        centred = pooled - pooled.mean()
        m2 = float(np.mean(centred**2))
        m4 = float(np.mean(centred**4))
        excess = m4 / m2**2 - 3.0 if m2 > 0 else float("nan")
    return {
        "sigma": sigma,
        "sigma_minutes": minutes,
        "unit": "rad/h" if observable == "offset" else "rad",
        "observable": observable,
        "repeats": repeats,
        "samples": int(pooled.size),
        "mean": float(np.mean(pooled)),
        "excess_kurtosis": excess,
    }


def probe(
    background: np.ndarray,
    n: int,
    hours: int,
    season: str,
    sigma: float,
    inject,
    *,
    observable: str = "offset",
    source: str = "nasa_power_2021",
    alpha: float,
    repeats: int = 100,
    seed: int = 11,
) -> dict[str, object]:
    """Run one injection ``repeats`` times and report what the detector saw.

    ``inject(series, r)`` returns ``(series, nodes)``. Two effects are reported,
    because they are not the same number and the difference is the whole point:

    ``read_gap``  what the instrument shows: the lying group's reading minus the
                  honest group's. This is the physical size of the lie.
    ``b_eff``     what the *statistic* sees: the lying node's residual against the
                  profiled mean of everybody else, ``t_k sigma sqrt((n-1)/n)``.
                  The GLRT can only use this, and it is the quantity the dilution
                  formula ``(n-k)/(n-1)`` is about. Even a single liar reads as
                  ``(n-1)/n`` of its gap, because the mean it is compared with
                  contains the liar itself; for a group the two factors combine.

    Reported besides them: both effects in minutes of clock, ``noncentrality``
    (the mean of ``t`` over the lying nodes -- what the power depends on), the
    share of parks flagged at ``alpha``, and the share where the flag landed on a
    lying node.
    """
    read_gaps, effects, stats = [], [], []
    flagged = localized = 0
    for r in range(repeats):
        park = build_park(background, n, hours, season, source=source, seed=seed + r)
        series, nodes = inject(park.series, r)
        x = read(series, hours, observable)
        honest = np.setdiff1d(np.arange(n), nodes)
        if observable == "phase":
            num = np.mean(np.exp(1j * x[nodes]))
            den = np.mean(np.exp(1j * x[honest]))
            read_gaps.append(float(np.angle(num / den)))
        else:
            read_gaps.append(float(x[nodes].mean() - x[honest].mean()))
        # Exactly the statistic the detector builds: no second implementation.
        t = detector.studentize(x / sigma)
        effects.append(float(np.mean(t[nodes]) * sigma * math.sqrt((n - 1.0) / n)))
        stats.append(float(np.mean(t[nodes])))
        verdict = detector.detect(x, sigma, alpha)
        flagged += int(verdict.flagged)
        localized += int(verdict.node in nodes)

    b = float(np.mean(effects))
    gap = float(np.mean(read_gaps))
    return {
        "observable": observable,
        "b_eff": b,
        "read_gap": gap,
        "b_eff_unit": "rad/h" if observable == "offset" else "rad",
        "b_eff_minutes": float(
            minutes_per_day(b) if observable == "offset" else minutes_from_phase(b)
        ),
        "read_gap_minutes": float(
            minutes_per_day(gap) if observable == "offset" else minutes_from_phase(gap)
        ),
        "b_eff_over_sigma": b / sigma if sigma else float("nan"),
        "noncentrality": float(np.mean(stats)),
        "detection_rate": flagged / repeats,
        "localisation_rate": localized / repeats,
        "repeats": repeats,
        "alpha": alpha,
        "sigma": sigma,
    }


# ──────────────────────────────────────────────────────────────────── grid ──


def load_criteria(path: Path | None = None) -> dict:
    """The criteria file, written before the run and hashed into the report."""
    path = path or BENCH_DIR / "criteria.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _key(*parts: object) -> str:
    return "|".join(str(p) for p in parts)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def unmeasurable(hours: int, observable: str) -> str | None:
    """Why this (reading, window) cell does not exist, or ``None`` if it does.

    Both refusals are declared limits of the *data*, not findings of the run: an
    hourly feed cannot give a frequency inside one revolution, and it cannot give
    a phase from two samples (one bit). Naming them here keeps the grid from
    printing a σ that measures the declaration instead of the instrument.
    """
    if observable == "offset" and hours < estimator.MIN_READ_HOURS:
        return (
            "частота не идентифицируема внутри одного оборота: "
            f"{hours} ч < {estimator.MIN_READ_HOURS:g} ч"
        )
    if observable == "phase" and int(hours) + 1 < estimator.MIN_PHASE_SAMPLES:
        return (
            f"фаза не идентифицируема на {int(hours) + 1} отсчётах "
            f"(нужно ≥ {estimator.MIN_PHASE_SAMPLES}): окно даёт знак изменения, "
            "а не фазу"
        )
    return None


def run_grid(criteria: dict, verbose: bool = True) -> dict:
    """Thresholds, measured noise and the resolution limit over the whole grid."""
    alpha = float(criteria["alpha"])
    target_power = float(criteria["target_power"])
    ns = tuple(int(v) for v in criteria["ns"])
    hours_grid = tuple(int(v) for v in criteria["hours"])
    seasons = tuple(criteria["seasons"])
    backgrounds = tuple(criteria["backgrounds"])
    repeats_sigma = int(criteria["repeats_sigma"])

    results: dict = {
        "criteria": criteria,
        "thresholds": {},
        "resolution_coefficient": {},
        "sigma": {},
        "resolution_limit": {},
        "injections": {},
        "roc": {},
    }

    for n in ns:
        tau = detector.null_threshold(n, alpha)
        results["thresholds"][str(n)] = {
            "alpha": alpha,
            "tau": tau,
            "tau_bonferroni": detector.bonferroni_threshold(n, alpha),
            # The threshold is a quantile of a Monte-Carlo sample, so it carries a
            # standard error of its own; a gap smaller than that is not a finding.
            "tau_se": detector.null_threshold_se(n, alpha),
            "draws": detector.DRAWS_NULL,
            "seed": detector.SEED,
        }
        results["resolution_coefficient"][str(n)] = detector.resolution_coefficient(
            n, alpha, target_power
        )
        if verbose:
            print(
                f"[grid] n={n:3d} tau={tau:.3f} "
                f"(Bonferroni {detector.bonferroni_threshold(n, alpha):.3f}) "
                f"c={results['resolution_coefficient'][str(n)]:.3f}",
                flush=True,
            )

    for bg_name in backgrounds:
        background = load_background(bg_name)
        for observable in OBSERVABLES:
            for season in seasons:
                for hours in hours_grid:
                    key = _key(bg_name, season, hours, observable)
                    reason = unmeasurable(hours, observable)
                    if reason is not None:
                        results["resolution_limit"][key] = {
                            "measurable": False,
                            "reason": reason,
                        }
                        if verbose:
                            print(
                                f"[skip ] {bg_name:15s} {observable:6s} "
                                f"{season:8s} T={hours:3d}h: {reason}",
                                flush=True,
                            )
                        continue
                    measured = measure_sigma(
                        background, max(ns), hours, season,
                        observable=observable, source=bg_name, repeats=repeats_sigma,
                    )
                    results["sigma"][key] = measured
                    if verbose:
                        print(
                            f"[sigma] {bg_name:15s} {observable:6s} {season:8s} "
                            f"T={hours:3d}h sigma={measured['sigma']:.6g} "
                            f"({measured['sigma_minutes']:.2f} min)",
                            flush=True,
                        )
                    for n in ns:
                        c = results["resolution_coefficient"][str(n)]
                        b_min = c * measured["sigma"]
                        tau = results["thresholds"][str(n)]["tau"]
                        results["resolution_limit"][_key(bg_name, season, hours, observable, n)] = {
                            "measurable": True,
                            "b_min": b_min,
                            "b_min_minutes": (
                                float(minutes_per_day(b_min))
                                if observable == "offset"
                                else float(minutes_from_phase(b_min))
                            ),
                            "b_min_over_sigma": c,
                            "b_min_analytic": detector.b_min_analytic(n, measured["sigma"], alpha, target_power),
                            "power_at_b_min": detector.power(b_min, measured["sigma"], n, tau),
                            "unit": "rad/h" if observable == "offset" else "rad",
                        }
    return results


def run_injections(results: dict, criteria: dict) -> dict:
    """Run the declared injections against both observables and record them."""
    alpha = float(criteria["alpha"])
    ns = tuple(int(v) for v in criteria["ns"])
    backgrounds = tuple(criteria["backgrounds"])
    repeats = int(criteria["repeats_inject"])
    primary = backgrounds[0]
    background = load_background(primary)

    injections = criteria["injections"]
    tau_hours = float(injections["timestamp_step"]["tau_hours"])
    factor = float(injections["level"]["factor"])
    ks = tuple(int(k) for k in injections["collusion"]["ks"])
    inject_seasons = tuple(injections.get("seasons", ("july",)))
    inject_hours = tuple(int(v) for v in injections.get("hours", (24, 168)))

    for observable in OBSERVABLES:
        for season in inject_seasons:
            for hours in inject_hours:
                key_sigma = _key(primary, season, hours, observable)
                if key_sigma not in results["sigma"]:
                    continue
                sigma = results["sigma"][key_sigma]["sigma"]
                t0 = hours // 2
                for n in ns:
                    def step(series, r, _t0=t0):
                        return inject_timestamp_step(series, [0], tau_hours, _t0), [0]

                    def control(series, r):
                        return inject_timestamp_step(series, [0], tau_hours, 0), [0]

                    def level(series, r):
                        return inject_level(series, [0], factor), [0]

                    def collude(series, r, _k=1, _t0=t0):
                        nodes = list(range(_k))
                        return inject_timestamp_step(series, nodes, tau_hours, _t0), nodes

                    cases = (
                        ("timestamp_step", step),
                        ("constant_offset", control),
                        ("level", level),
                    )
                    for name, fn in cases:
                        results["injections"][_key(name, primary, season, hours, n, observable)] = probe(
                            background, n, hours, season, sigma, fn,
                            observable=observable, source=primary, alpha=alpha, repeats=repeats,
                        )
                    for k in ks:
                        if k > n:
                            continue
                        results["injections"][_key("collusion", primary, season, hours, n, observable, k)] = probe(
                            background, n, hours, season, sigma,
                            lambda s, r, _k=k, _t0=t0: collude(s, r, _k, _t0),
                            observable=observable, source=primary, alpha=alpha, repeats=repeats,
                        )
                print(f"[inject] {observable:6s} {primary} {season:8s} T={hours:3d}h done", flush=True)
    return results


def run_roc(results: dict, criteria: dict) -> dict:
    """ROC curves: one family over ``n``, one over the window length, per reading.

    The lying node is given ``b = c(n) * sigma`` -- the resolution limit of §6, so
    each curve is drawn through its own operating point (power ``target_power`` at
    the declared ``alpha``) rather than far above it, where every curve would sit
    at the top-left corner and the plot would say nothing. Because ``b`` is stated
    in units of ``sigma``, the window-length family is identical by construction:
    that invariance is the point of drawing it.
    """
    ns = tuple(int(v) for v in criteria["ns"])
    hours_grid = tuple(int(v) for v in criteria["hours"])
    primary = tuple(criteria["backgrounds"])[0]
    season = str(criteria.get("roc_season", "july"))
    hours = int(criteria.get("roc_hours", 24))
    alpha = float(criteria["alpha"])
    target_power = float(criteria["target_power"])
    results.setdefault("roc", {})

    for observable in OBSERVABLES:
        key = _key(primary, season, hours, observable)
        if key in results["sigma"]:
            sigma = results["sigma"][key]["sigma"]
            for n in ns:
                c = detector.resolution_coefficient(n, alpha, target_power)
                results["roc"][_key("n", n, observable, hours)] = detector.roc(
                    n, sigma, c * sigma
                )
        for h in hours_grid:
            key_h = _key(primary, season, h, observable)
            if key_h not in results["sigma"]:
                continue
            sigma_h = results["sigma"][key_h]["sigma"]
            n0 = ns[0]
            c = detector.resolution_coefficient(n0, alpha, target_power)
            results["roc"][_key("hours", h, observable, n0)] = detector.roc(
                n0, sigma_h, c * sigma_h
            )
    return results


def _coalition_rows(
    n: int, k: int, b: float, sigma: float, draws: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(statistic, node)`` for ``draws`` parks where nodes ``0..k-1`` lie by ``b``.

    The model of ``detector`` written down once: ``x = mu 1 + b 1_S + sigma z``,
    with ``mu`` profiled out by the statistic itself. This is M2's noise model --
    the real background enters through ``sigma``, measured in M1.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((draws, n)) * sigma
    if 0 < k <= n:
        x[:, :k] += b
    return detector.statistics_and_nodes(x / sigma)


def _coalition_power(
    n: int, k: int, b: float, sigma: float, tau: float, draws: int, seed: int
) -> tuple[float, float | None]:
    """Power against a coalition of ``k``, and the share of flags landing on honest.

    The second number is what makes the inversion visible: the *value* of the
    statistic says "something is wrong", the *node* says "this one" -- and for
    ``k > n/2`` that node is honest, with probability one.
    """
    stats, nodes = _coalition_rows(n, k, b, sigma, draws, seed)
    hit = stats > tau
    power = float(np.mean(hit))
    if not hit.any() or k >= n:
        return power, None
    return power, float(np.mean(nodes[hit] >= k))


def _coalition_b_min(
    n: int,
    k: int,
    sigma: float,
    tau: float,
    target_power: float,
    draws: int,
    seed: int,
    rtol: float = 2e-3,
) -> float:
    """Bisection on the coalition's Monte Carlo power: the measured ``b_min(k)``."""

    def reached(b: float) -> bool:
        return _coalition_power(n, k, b, sigma, tau, draws, seed)[0] >= target_power

    hi = max(4.0 * sigma * math.sqrt(max(n - 1, 1)), sigma * 1e-3)
    while not reached(hi):
        hi *= 2.0
        if hi > 1e7 * sigma:
            return math.inf
    lo = 0.0
    while hi - lo > rtol * max(hi, 1e-30):
        mid = 0.5 * (lo + hi)
        if reached(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def load_criteria_m2(path: Path | None = None) -> dict:
    """The M2 pre-registration: predictions first, measurements second."""
    path = path or BENCH_DIR / "criteria_m2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_theorem(results: dict, criteria_m2: dict, verbose: bool = True) -> dict:
    """M2: check the theorem's closed-form predictions, on noise and in the field.

    Three groups of checks, none of which can be tuned:

    ``algebra``   the identities (gauge, kernel, orientation), evaluated on random
                  vectors and on real readings -- an identity holds or the theorem
                  is false;
    ``measured``  the noise model at the measured ``sigma``: power, the node the
                  flag lands on, and ``b_min(k)`` by bisection, against
                  ``theorem.b_min_ratio``;
    ``real``      the same coalitions injected into the real background, where the
                  predictions must hold too, plus the kernel in the field (a full
                  coalition at growing ``b`` must not raise the alarm rate) and the
                  anchored test that fixes the gauge.

    Every number is stored, including the ones that come out ugly: a criterion
    that fails is a result, and the report says so.
    """
    import hashlib

    from . import theorem

    alpha = float(criteria_m2["alpha"])
    target = float(criteria_m2["target_power"])
    draws = int(criteria_m2["noise_draws"])
    seed = int(criteria_m2["noise_seed"])
    ns = [int(v) for v in criteria_m2["ns"]]
    ks_by_n = {int(k): [int(v) for v in vals] for k, vals in criteria_m2["ks"].items()}

    block: dict[str, object] = {
        "alpha": alpha,
        "target_power": target,
        "criteria": criteria_m2,
        "criteria_sha256": hashlib.sha256(
            (BENCH_DIR / "criteria_m2.json").read_bytes()
        ).hexdigest(),
    }

    # ── the identities ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 40)
    t0 = detector.studentize(x)
    gauge_drift = max(
        float(np.max(np.abs(detector.studentize(x + c) - t0)))
        for c in (-1e3, -1.0, 1.0, 1e3)
    )
    kernel_statistic = 0.0
    for n in ns:
        for b in (1e-6, 1.0, 1e6):
            kernel_statistic = max(kernel_statistic, abs(detector.statistic(np.full(n, b))))

    orientation = {
        "reading_gap": 0.0,
        "statistic_gap": 0.0,
        "same_node": True,
        "same_verdict": True,
        "cells": 0,
    }
    for n in ns:
        for k in ks_by_n[n]:
            if not 0 < k < n:
                continue
            x_a, x_b, _, _ = theorem.two_worlds(n, k, 0.02 * math.sqrt(n))
            orientation["reading_gap"] = max(
                orientation["reading_gap"], float(np.max(np.abs(x_a - x_b)))
            )
            orientation["statistic_gap"] = max(
                orientation["statistic_gap"],
                float(np.max(np.abs(detector.studentize(x_a) - detector.studentize(x_b)))),
            )
            va = detector.detect(x_a, sigma=1.0, alpha=alpha)
            vb = detector.detect(x_b, sigma=1.0, alpha=alpha)
            orientation["same_node"] &= va.node == vb.node
            orientation["same_verdict"] &= (va.flagged, va.node) == (vb.flagged, vb.node)
            orientation["cells"] += 1

    block["algebra"] = {
        "gauge_drift": gauge_drift,
        "kernel_statistic": kernel_statistic,
        "orientation": orientation,
    }

    # ── predictions, then the noise model ─────────────────────────────────
    block["predicted"] = theorem.predicted_rows(ns, ks_by_n)

    measured: list[dict[str, object]] = []
    for n in ns:
        tau = detector.null_threshold(n, alpha)
        b1 = detector.b_min(n, 1.0, alpha, target)  # in units of sigma
        for k in ks_by_n[n]:
            if k == n:
                stats, _ = _coalition_rows(n, k, 1e3, 1.0, draws, seed)
                measured.append(
                    {
                        "n": n,
                        "k": k,
                        "b1_over_sigma": b1,
                        "b_min_over_sigma": math.inf,
                        "b_min_ratio_predicted": math.inf,
                        "b_min_ratio_measured": math.inf,
                        "power_at_b1": float(np.mean(stats > tau)),
                        "statistic_at_huge_b": float(np.max(stats)),
                        "flag_on_honest": None,
                        "flagged_side": theorem.flagged_side(n, k),
                    }
                )
                continue
            b_min_k = _coalition_b_min(n, k, 1.0, tau, target, draws, seed)
            power, honest_share = _coalition_power(n, k, b1, 1.0, tau, draws, seed)
            # The chord test, when the coalition is known in advance: same b, same
            # model, one contrast instead of n studentised nodes.
            liars = np.zeros(n, dtype=bool)
            liars[:k] = True
            rng_c = np.random.default_rng(seed + 7)
            xc = rng_c.standard_normal((draws, n))
            xc[:, :k] += b1
            diff = xc[:, liars].mean(axis=1) - xc[:, ~liars].mean(axis=1)
            zc = diff / math.sqrt(1.0 / k + 1.0 / (n - k))
            tau_c = NormalDist().inv_cdf(1.0 - alpha / 2.0)
            power_chord = float(np.mean(np.abs(zc) > tau_c))
            # At a strong signal the *identity* of the flagged node is the claim;
            # at b_min the statistic is barely over the threshold and the two
            # groups mix, so that share is reported but not asserted.
            _, honest_share_strong = _coalition_power(
                n, k, 3.0 * b_min_k, 1.0, tau, draws, seed
            )
            measured.append(
                {
                    "n": n,
                    "k": k,
                    "b1_over_sigma": b1,
                    "b_min_over_sigma": b_min_k,
                    "b_min_ratio_predicted": theorem.b_min_ratio(n, k),
                    "b_min_ratio_measured": b_min_k / b1,
                    "bound_holds": bool(
                        b_min_k / b1
                        <= theorem.b_min_ratio(n, k)
                        * (1.0 + float(block["criteria"]["tolerances"]["b_min_ratio_relative"]))
                    ),
                    "power_at_b1": power,
                    "power_chord_at_b1": power_chord,
                    "power_at_measured_b_min": _coalition_power(
                        n, k, b_min_k, 1.0, tau, draws, seed
                    )[0],
                    "flag_on_honest": honest_share,
                    "flag_on_honest_strong": honest_share_strong,
                    "flagged_side": theorem.flagged_side(n, k),
                }
            )
        if verbose:
            print(f"[theorem] noise model n={n} done", flush=True)
    block["measured"] = measured

    # ── the field: the same coalitions, on the real background ────────────
    real_cfg = criteria_m2["real"]
    primary = "nasa_power_2021"
    background = load_background(primary)
    real: list[dict[str, object]] = []
    for season in real_cfg["seasons"]:
        for hours in real_cfg["hours"]:
            for observable in real_cfg["observables"]:
                key_sigma = _key(primary, season, int(hours), observable)
                if key_sigma not in results.get("sigma", {}):
                    continue
                sigma = results["sigma"][key_sigma]["sigma"]
                b = float(real_cfg["b_over_sigma"]) * sigma
                for n in ns:
                    for k in ks_by_n[n]:
                        def collude(series, r, _k=k, _b=b):
                            nodes = list(range(_k))
                            return inject_clock_rate(series, nodes, _b), nodes

                        cell = probe(
                            background,
                            n,
                            int(hours),
                            season,
                            sigma,
                            collude,
                            observable=observable,
                            source=primary,
                            alpha=alpha,
                            repeats=int(real_cfg["repeats"]),
                            seed=seed + 17,
                        )
                        cell.update(
                            {
                                "n": n,
                                "k": k,
                                "season": season,
                                "hours": int(hours),
                                "b_over_sigma": float(real_cfg["b_over_sigma"]),
                                "flagged_side_predicted": theorem.flagged_side(n, k),
                                "b_min_ratio_predicted": theorem.b_min_ratio(n, k),
                                "singular": bool(k == n),
                            }
                        )
                        real.append(cell)
                if verbose:
                    print(
                        f"[theorem] field {season} T={hours}h {observable} done", flush=True
                    )
    block["real"] = real

    # ── the kernel in the field: everybody lies, nothing happens ──────────
    kcfg = criteria_m2["kernel_real"]
    key_sigma = _key(primary, kcfg["season"], int(kcfg["hours"]), kcfg["observable"])
    kernel_real: list[dict[str, object]] = []
    if key_sigma in results.get("sigma", {}):
        sigma = results["sigma"][key_sigma]["sigma"]
        n = int(kcfg["n"])
        for b_over in kcfg["b_over_sigma"]:
            b_all = float(b_over) * sigma

            def all_lie(series, r, _b=b_all, _n=n):
                nodes = list(range(_n))
                return inject_clock_rate(series, nodes, _b), nodes

            cell = probe(
                background,
                n,
                int(kcfg["hours"]),
                kcfg["season"],
                sigma,
                all_lie,
                observable=kcfg["observable"],
                source=primary,
                alpha=alpha,
                repeats=int(kcfg["repeats"]),
                seed=seed + 29,
            )
            cell.update({"n": n, "k": n, "b_over_sigma": float(b_over)})
            kernel_real.append(cell)
        if verbose:
            print("[theorem] kernel in the field done", flush=True)
    block["kernel_real"] = kernel_real

    # ── the converse: one anchor, and what it costs ───────────────────────
    anchor: list[dict[str, object]] = []
    tau_a = NormalDist().inv_cdf(1.0 - alpha)
    for n in ns:
        b1 = detector.b_min(n, 1.0, alpha, target)
        for k in ks_by_n[n]:
            anchored = theorem.anchor_b_min(n, 1.0, alpha, target, k=k)
            unanchored = math.inf if k == n else b1 * theorem.b_min_ratio(n, k)
            # Measured: the anchored test on the noise model, k liars past the anchor.
            rng_a = np.random.default_rng(seed + 41)
            y = rng_a.normal(0.0, 1.0, (draws // 4, n - 1))
            if 0 < k < n:
                y[:, :k] += anchored
            z = math.sqrt(n - 1) * y.mean(axis=1)
            anchor.append(
                {
                    "n": n,
                    "k": k,
                    "anchored_b_min_over_sigma": anchored,
                    "unanchored_b_min_over_sigma": unanchored,
                    "gain": (
                        unanchored / anchored
                        if math.isfinite(anchored) and math.isfinite(unanchored)
                        else (math.inf if math.isfinite(anchored) else None)
                    ),
                    "anchored_power_measured": float(np.mean(z > tau_a)),
                    "anchored_power_analytic": theorem.anchor_power(
                        anchored, 1.0, n, alpha, k=k
                    )
                    if math.isfinite(anchored)
                    else 0.0,
                }
            )
        anchor.append(
            {
                "n": n,
                "crossover_k": theorem.anchor_crossover(n, alpha, target),
                "sqrt_n": math.sqrt(n),
            }
        )
    block["anchor"] = anchor

    results["theorem"] = block
    return results


def run_all(criteria: dict | None = None, verbose: bool = True) -> dict:
    """Everything the report needs, in one deterministic pass."""
    criteria = criteria or load_criteria()
    results = run_grid(criteria, verbose=verbose)
    results = run_injections(results, criteria)
    results = run_roc(results, criteria)
    m2_path = BENCH_DIR / "criteria_m2.json"
    if m2_path.exists():
        results = run_theorem(results, load_criteria_m2(m2_path), verbose=verbose)
    return results


def save_results(results: dict, path: Path | None = None) -> Path:
    """Write ``bench/results.json`` (numpy types flattened, order preserved)."""
    path = path or BENCH_DIR / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path




