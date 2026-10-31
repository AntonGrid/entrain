"""GLRT for one lying node in a park: the math, and nothing about hardware.

The null and the alternative
----------------------------
``x[i]`` is producer ``i``'s frequency offset against the park clock, averaged
over a window (``model.Observation``). Honest producers share a common mean
``mu`` (the park's own reading of the clock) and scatter around it::

    H0:  x_i ~ N(mu, sigma^2)                 i = 1..n
    H1:  x_k ~ N(mu + b, sigma^2)             for exactly one unknown k, b != 0
         x_i ~ N(mu, sigma^2)                 for i != k

``sigma`` is measured on the background, never guessed by the detector.

The test statistic
------------------
``mu`` and ``b`` are nuisance parameters, so both are profiled out. For a
candidate ``k`` the profile MLEs are ``mu_hat = mean_{i != k} x_i`` and
``b_hat = x_k - mean_{i != k} x_i``, which gives the generalized likelihood
ratio::

    Lam_k = b_hat^2 / (2 sigma^2 (1 + 1/(n-1)))
    Lam   = max_k Lam_k  =  (1/2) * max_k t_k^2

with the studentized residual, in closed form (no masking of one row)::

    t_k = (n x_k - sum_i x_i) / sqrt(n (n-1))

Reject H0 when ``max_k |t_k| > tau``. Under H0, ``t`` is exactly multivariate
normal with unit variances and correlation ``-1/(n-1)`` between distinct
components (this is what makes a naive ``z``-test wrong: the residuals are
correlated and the maximum over ``n`` of them is not a single normal). The
distribution of ``max_k |t_k|`` therefore depends on ``n`` only -- not on
``sigma``, not on ``mu`` -- so one table of quantiles per ``n`` serves every
park. Those quantiles are computed by Monte Carlo with a declared seed, and
cross-checked against the Bonferroni bound ``n * 2 * (1 - Phi(tau))``.

The operator chooses ``alpha``, the declared false-alarm risk per window, and
the threshold follows. There is no magic 0.5 anywhere in this module.

Power and the minimum detectable offset
---------------------------------------
Under H1 the offending node's statistic is exactly ``N(delta, 1)`` with
``delta = b / (sigma * sqrt(1 + 1/(n-1)))``, and an honest node's is
``N(-delta/(n-1), 1)``. That second mean is the collusion dilution, stated as a
formula: if ``k`` nodes lie together the offending node's mean becomes
``b (n-k) / (n-1)``, so a coherent group of ``n`` is invisible by construction.

``b_min`` solves ``power(b_min) = 1 - beta`` at the declared ``alpha``::

    b_min = (tau_alpha + z_{1-beta}) * sigma * sqrt(1 + 1/(n-1))     (analytic)
    b_min = bisection of the Monte Carlo power                     (exact)

The analytic form ignores the small chance that an honest node trips the
threshold; the report prints both and the difference.

Past half the park
-------------------
The same algebra, pushed one step further, is the M2 theorem (``theorem``): the
park sees only contrasts, so a lie enters through its chord
``sqrt(k(n-k)/n)``, a coalition larger than ``n/2`` puts the *honest* minority at
the top of ``|t|``, and the orientation of a split is not in the data at all.
Nothing in this module changes because of it -- the test is still the test of one
liar -- but the reading of its output does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Default Monte Carlo effort. Declared, so that a number can be reproduced.
DRAWS_NULL: int = 200_000
DRAWS_POWER: int = 20_000
SEED: int = 20261030


# ────────────────────────────────────────────────────────────── statistics ──


def studentize(x: np.ndarray) -> np.ndarray:
    """``t_k = (n x_k - sum x) / sqrt(n (n-1))`` for every ``k``."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least two producers")
    return (n * x - x.sum()) / math.sqrt(n * (n - 1))


def statistic(x: np.ndarray) -> float:
    """``max_k |t_k|`` -- the GLRT statistic up to the factor 1/2."""
    return float(np.max(np.abs(studentize(x))))


def studentize_rows(x: np.ndarray) -> np.ndarray:
    """``t_k`` for every row of a ``(draws, n)`` matrix: the same closed form.

    The row-wise form of ``studentize``, so that a bench measuring a thousand
    parks does not grow a second implementation of the statistic.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[1]
    if n < 2:
        raise ValueError("need at least two producers")
    return (n * x - x.sum(axis=1, keepdims=True)) / math.sqrt(n * (n - 1))


def statistics_and_nodes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``max_k |t_k|`` and the node it points at, row-wise.

    The node matters as much as the value: for a coalition larger than half the
    park the maximum belongs to an *honest* node (``theorem.flagged_side``), so a
    bench that records only the statistic cannot see the inversion at all.
    """
    t = np.abs(studentize_rows(x))
    node = np.argmax(t, axis=1)
    return t[np.arange(t.shape[0]), node], node


def _statistics_rows(x: np.ndarray) -> np.ndarray:
    """``max_k |t_k|`` for each row of a ``(draws, n)`` matrix."""
    return statistics_and_nodes(x)[0]


def log_likelihood_ratio(x: np.ndarray, sigma: float) -> tuple[float, int]:
    """The GLRT statistic and the node it points at, in nats."""
    t = studentize(np.asarray(x, dtype=float) / sigma)
    k = int(np.argmax(np.abs(t)))
    return 0.5 * t[k] ** 2, k


# ──────────────────────────────────────────────────────────── null table ──


@lru_cache(maxsize=64)
def _null_sample(n: int, draws: int, seed: int) -> tuple[float, ...]:
    """Sorted ``max_k |t_k|`` under H0, chunked to keep memory flat."""
    rng = np.random.default_rng(seed)
    out = np.empty(draws, dtype=np.float64)
    chunk = 20_000
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        z = rng.standard_normal((stop - start, n))
        out[start:stop] = _statistics_rows(z)
    out.sort()
    return tuple(out.tolist())


def null_threshold(
    n: int, alpha: float, draws: int = DRAWS_NULL, seed: int = SEED
) -> float:
    """``tau_alpha``: the ``1 - alpha`` quantile of ``max_k |t_k|`` under H0.

    ``alpha`` is the operator's declared risk per window: the share of honest
    parks of size ``n`` that will be flagged. Monte Carlo, seed declared.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    sample = _null_sample(n, draws, seed)
    idx = min(len(sample) - 1, int(math.ceil((1.0 - alpha) * len(sample))) - 1)
    return float(sample[max(0, idx)])


def null_pvalue(
    observed: float, n: int, draws: int = DRAWS_NULL, seed: int = SEED
) -> float:
    """Empirical p-value of an observed statistic under H0 (never exactly 0)."""
    sample = np.asarray(_null_sample(n, draws, seed))
    hits = int(np.count_nonzero(sample >= observed))
    return (1.0 + hits) / (1.0 + sample.size)


def null_threshold_se(n: int, alpha: float, draws: int = DRAWS_NULL, seed: int = SEED) -> float:
    """Monte-Carlo standard error of ``null_threshold``, in units of the statistic.

    The threshold is an order statistic, so it carries the binomial noise of the
    quantile, scaled by the local spacing of the null distribution: half the
    distance between the order statistics one sampling error either side of the
    target. It is reported next to the threshold, because a bound that holds only
    up to two decimals is a bound only up to two decimals.
    """
    sample = _null_sample(n, draws, seed)
    se_p = math.sqrt(alpha * (1.0 - alpha) / draws)
    m = max(1, int(round(1.96 * se_p * draws)))
    centre = min(draws - 1, max(0, int(math.ceil((1.0 - alpha) * draws)) - 1))
    lo = max(0, centre - m)
    hi = min(draws - 1, centre + m)
    return float((sample[hi] - sample[lo]) / (2.0 * 1.96))


def bonferroni_threshold(n: int, alpha: float) -> float:
    """``Phi^{-1}(1 - alpha/(2n))`` -- the bound Monte Carlo is checked against."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * n))


# ───────────────────────────────────────────────────────────── power, b_min ──


def _alternative_rows(
    n: int, sigma: float, b: float, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Statistics under H1 with the lie at node 0 (exchangeable, so no loss).

    The physical park is ``sigma * z`` with a lie ``b`` added at one node, and the
    detector standardizes by ``sigma`` before comparing to ``tau`` (see
    ``detect``). The statistic therefore depends on ``b / sigma`` alone -- which
    is the whole reason ``b_min`` is proportional to the measured noise and one
    coefficient ``c(n)`` serves every season.
    """
    x = rng.standard_normal((draws, n)) * sigma
    x[:, 0] += b
    return _statistics_rows(x / sigma)


def power(
    b: float,
    sigma: float,
    n: int,
    tau: float,
    draws: int = DRAWS_POWER,
    seed: int = SEED,
) -> float:
    """Monte Carlo power of the test at threshold ``tau`` for offset ``b``."""
    rng = np.random.default_rng(seed + 1)
    stats = _alternative_rows(n, sigma, b, draws, rng)
    return float(np.mean(stats > tau))


def power_analytic(b: float, sigma: float, n: int, tau: float) -> float:
    """``P(|N(delta,1)| > tau)``: the offending node alone, honest nodes dropped."""
    from statistics import NormalDist

    nd = NormalDist()
    delta = abs(b) / (sigma * math.sqrt(1.0 + 1.0 / (n - 1)))
    return float(nd.cdf(delta - tau) + nd.cdf(-delta - tau))


def b_min_analytic(
    n: int, sigma: float, alpha: float, target_power: float = 0.9
) -> float:
    """``(tau_alpha + z_{1-beta}) * sigma * sqrt(1 + 1/(n-1))``."""
    from statistics import NormalDist

    tau = null_threshold(n, alpha)
    z = NormalDist().inv_cdf(target_power)
    return float((tau + z) * sigma * math.sqrt(1.0 + 1.0 / (n - 1)))


def b_min(
    n: int,
    sigma: float,
    alpha: float,
    target_power: float = 0.9,
    draws: int = DRAWS_POWER,
    seed: int = SEED,
    rtol: float = 1e-3,
) -> float:
    """Exact ``b_min``: bisection on the Monte Carlo power (monotone in ``b``)."""
    tau = null_threshold(n, alpha)
    analytic = b_min_analytic(n, sigma, alpha, target_power)

    def reached(b: float) -> bool:
        return power(b, sigma, n, tau, draws, seed) >= target_power

    lo, hi = 0.0, max(analytic * 2.0, sigma * 1e-3)
    while not reached(hi):
        hi *= 2.0
        if hi > 1e6 * sigma:
            raise RuntimeError("power never reaches the target -- check the test")
    while hi - lo > rtol * max(hi, 1e-30):
        mid = 0.5 * (lo + hi)
        if reached(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


@lru_cache(maxsize=64)
def resolution_coefficient(
    n: int,
    alpha: float,
    target_power: float = 0.9,
    draws: int = DRAWS_POWER,
    seed: int = SEED,
) -> float:
    """``c(n) = b_min / sigma``: b_min is linear in ``sigma``, by construction.

    The whole test is scale invariant -- multiplying every offset and ``sigma``
    by a constant multiplies the statistic's distribution by the same constant
    -- so one coefficient per ``n`` serves every season's measured noise. The
    linearity is exact in the mathematics and holds to the bisection tolerance
    in the Monte Carlo.
    """
    return b_min(n, 1.0, alpha, target_power, draws, seed)


# ───────────────────────────────────────────────────────────────────── ROC ──


def roc(
    n: int,
    sigma: float,
    b: float,
    alphas: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3),
    draws: int = 40_000,
    seed: int = SEED,
) -> dict[str, object]:
    """ROC of the GLRT: declared risks on the x axis, measured power on the y.

    For each ``alpha`` the threshold is the null quantile, the false-alarm rate
    is *measured* on fresh null draws (so ``alpha`` is verified, not restated),
    and the power is measured against a single lying node. Returns the rows, the
    empirical curve, its partial area, and ``b``/``sigma`` to label the plot.

    ``auc`` is the area **normalised by the covered false-alarm range**: the
    declared grid stops at the largest ``alpha``, so the conventional whole-axis
    AUC is not available here and quoting a raw trapezoid would understate a
    perfectly separated detector. ``auc_range`` says what was covered.
    """
    rng = np.random.default_rng(seed + 2)
    null_stats = _statistics_rows(rng.standard_normal((draws, n)))
    alt_stats = _alternative_rows(n, sigma, b, draws, rng)

    rows = []
    for alpha in alphas:
        tau = null_threshold(n, alpha)
        rows.append(
            {
                "alpha": alpha,
                "tau": tau,
                "fpr": float(np.mean(null_stats > tau)),
                "tpr": float(np.mean(alt_stats > tau)),
                "tpr_analytic": power_analytic(b, sigma, n, tau),
            }
        )
    fpr = np.array([r["fpr"] for r in rows], dtype=float)
    tpr = np.array([r["tpr"] for r in rows], dtype=float)
    order = np.argsort(fpr)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x / 2.x
    span = float(fpr.max() - fpr.min()) if fpr.size > 1 else 0.0
    # The declared risk grid covers only part of the false-alarm axis, so the raw
    # area under the sampled points is *not* the conventional AUC -- it is
    # bounded by the largest alpha in the grid. Normalising by the covered range
    # turns it back into "1 = the two readings are perfectly separated here".
    auc = float(trapz(tpr[order], fpr[order]) / span) if span > 1e-12 else float("nan")
    return {
        "rows": rows,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "auc_range": [float(fpr.min()), float(fpr.max())],
        "b": b,
        "sigma": sigma,
    }


# ────────────────────────────────────────────────────────────── decisions ──


@dataclass(frozen=True)
class Verdict:
    """What the detector says about one park, at one declared risk."""

    n: int
    alpha: float
    tau: float
    statistic: float
    p_value: float
    node: int
    flagged: bool
    b_hat: float
    sigma: float

    @property
    def b_hat_minutes_per_day(self) -> float:
        from .model import minutes_per_day

        return float(minutes_per_day(self.b_hat))

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "alpha": self.alpha,
            "tau": self.tau,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "node": self.node,
            "flagged": self.flagged,
            "b_hat_rad_per_h": self.b_hat,
            "b_hat_minutes_per_day": self.b_hat_minutes_per_day,
            "sigma_rad_per_h": self.sigma,
        }


def detect(x: np.ndarray, sigma: float, alpha: float) -> Verdict:
    """Run the GLRT on one park at the operator's declared risk ``alpha``."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    stat = statistic(x / sigma)
    tau = null_threshold(n, alpha)
    t = studentize(x / sigma)
    k = int(np.argmax(np.abs(t)))
    b_hat = float(x[k] - np.delete(x, k).mean())
    return Verdict(
        n=n,
        alpha=alpha,
        tau=tau,
        statistic=stat,
        p_value=null_pvalue(stat, n),
        node=k,
        flagged=bool(stat > tau),
        b_hat=b_hat,
        sigma=sigma,
    )


def collusion_dilution(n: int, k: int) -> float:
    """How much of a lie survives when ``k`` nodes lie together: (n-k)/(n-1)."""
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n")
    if n == 1:
        return 0.0
    return (n - k) / (n - 1)


def alpha_for(verdict_statistic: float, n: int) -> float:
    """The risk at which this statistic would *just* be flagged (the p-value).

    This is the inverse question the operator actually asks: not "is 0.63 more
    than 0.5", but "what false-alarm risk am I accepting if I react to this".
    """
    return null_pvalue(verdict_statistic, n)

