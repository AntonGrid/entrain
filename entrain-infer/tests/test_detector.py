"""The GLRT does what its documentation claims, and nothing more.

These are the guard rails for M1: closed-form statistic, size equal to the
declared alpha, power at b_min equal to the declared target, exact scale
invariance, and the collusion dilution that M2 will have to prove.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from entrain_infer import detector


def test_statistic_matches_profile_likelihood() -> None:
    """``max_k |t_k|`` is the profile likelihood ratio of the documented H1."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=7)
    sigma = 2.5
    # brute force: for each candidate k, profile mu over the honest ones
    best = 0.0
    for k in range(x.size):
        honest = np.delete(x, k)
        mu = honest.mean()
        b = x[k] - mu
        var = sigma**2
        lam = b**2 / (2.0 * var * (1.0 + 1.0 / (x.size - 1)))
        best = max(best, lam)
    assert detector.log_likelihood_ratio(x, sigma)[0] == pytest.approx(best, rel=1e-12)


def test_studentize_is_orthogonal_to_the_park_mean() -> None:
    """Adding a constant to every producer must not move the statistic."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=11)
    t = detector.studentize(x)
    assert float(t.sum()) == pytest.approx(0.0, abs=1e-12)
    assert detector.statistic(x) == pytest.approx(detector.statistic(x + 17.0), rel=1e-12)


def test_threshold_is_the_declared_risk() -> None:
    """The empirical false-alarm rate at tau is the declared alpha."""
    n, alpha = 20, 1e-3
    tau = detector.null_threshold(n, alpha)
    rng = np.random.default_rng(3)
    stats = np.array([detector.statistic(rng.normal(size=n)) for _ in range(20_000)])
    rate = float(np.mean(stats > tau))
    assert 0.4 * alpha <= rate <= 2.5 * alpha


def test_threshold_agrees_with_bonferroni_up_to_monte_carlo_noise() -> None:
    """The Bonferroni bound is conservative; the gap must not exceed its own error."""
    for n in (5, 10, 30, 100):
        tau = detector.null_threshold(n, 1e-3)
        se = detector.null_threshold_se(n, 1e-3)
        bound = detector.bonferroni_threshold(n, 1e-3)
        assert tau <= bound + se
        assert bound - tau < 0.15
        assert 0.0 < se < 0.02


def test_threshold_error_shrinks_with_more_draws() -> None:
    """More Monte Carlo, tighter threshold -- so the reported gap is measurable."""
    coarse = detector.null_threshold_se(10, 1e-3, draws=20_000)
    fine = detector.null_threshold_se(10, 1e-3, draws=200_000)
    assert fine < coarse


def test_pvalue_never_zero() -> None:
    """An extreme observation gets a small, not a zero, p-value."""
    p = detector.null_pvalue(50.0, 10)
    assert 0.0 < p < 1e-4


def test_scale_invariance() -> None:
    """b_min is linear in sigma: the whole test is scale invariant."""
    n, alpha = 10, 1e-3
    tau = detector.null_threshold(n, alpha)
    b1 = detector.b_min(n, 1.0, alpha, 0.9, draws=4000, rtol=1e-7)
    b2 = detector.b_min(n, 2.0, alpha, 0.9, draws=4000, rtol=1e-7)
    assert b2 == pytest.approx(2.0 * b1, rel=1e-5)
    assert detector.power(b1, 1.0, n, tau) == pytest.approx(0.9, abs=0.02)


def test_power_at_b_min_reaches_the_target() -> None:
    """The exact b_min gives the exact target power."""
    n, alpha, target = 30, 1e-2, 0.8
    tau = detector.null_threshold(n, alpha)
    b = detector.b_min(n, 1.0, alpha, target, draws=6000)
    assert detector.power(b, 1.0, n, tau, draws=6000) >= target - 1e-9


def test_analytic_b_min_is_conservative() -> None:
    """The closed form ignores honest nodes crossing tau, so it cannot be exact.

    Ignoring that extra evidence makes the closed form ask for a slightly larger
    offset than the Monte Carlo needs -- it is conservative, not optimistic.
    """
    n, alpha = 10, 1e-3
    exact = detector.b_min(n, 1.0, alpha, 0.9, draws=8000)
    analytic = detector.b_min_analytic(n, 1.0, alpha, 0.9)
    assert exact <= analytic
    assert (analytic - exact) / exact < 0.05
    assert detector.power_analytic(analytic, 1.0, n, detector.null_threshold(n, alpha)) == (
        pytest.approx(0.9, abs=1e-9)
    )


def test_collusion_dilution_in_the_statistic() -> None:
    """k identical liars show (n-k)/(n-1) of the shift, exactly as documented."""
    n, k, b = 51, 3, 1.0
    x = np.zeros(n)
    x[:k] = b  # the whole group lies by the same amount
    t = detector.studentize(x)
    expected = b * (n - k) / math.sqrt(n * (n - 1))
    assert float(t[0]) == pytest.approx(expected, rel=1e-9)
    assert detector.collusion_dilution(n, k) == pytest.approx((n - k) / (n - 1), rel=1e-12)


def test_full_collusion_is_invisible() -> None:
    """If everyone lies together there is no park left to tell the truth."""
    n, b = 12, 3.0
    x = np.full(n, b)
    assert detector.statistic(x) == pytest.approx(0.0, abs=1e-12)
    assert not detector.detect(x, 1.0, 1e-3).flagged


def test_detect_localises_the_liar() -> None:
    """At a loud offset the flag lands on the guilty node."""
    rng = np.random.default_rng(4)
    n = 30
    x = rng.normal(0.0, 1.0, n)
    x[7] += 6.0
    verdict = detector.detect(x, 1.0, 1e-3)
    assert verdict.flagged
    assert verdict.node == 7
    assert verdict.p_value <= 1e-3
    assert verdict.b_hat == pytest.approx(6.0, rel=0.2)


def test_roc_is_monotone_and_reports_the_operating_point() -> None:
    """Power never falls as the declared risk grows, and the rows carry alpha."""
    curve = detector.roc(10, 1.0, 6.0, draws=16_000)
    tpr = list(curve["tpr"])
    assert all(b >= a - 1e-9 for a, b in zip(tpr, tpr[1:]))
    assert curve["auc"] > 0.9
    alpha = 1e-3
    row = next(r for r in curve["rows"] if abs(r["alpha"] - alpha) < 1e-12)
    assert 0.3 * alpha <= row["fpr"] <= 3.0 * alpha
    assert row["tpr"] > 0.9
    assert row["tau"] == pytest.approx(detector.null_threshold(10, alpha))
