"""M2: the theorem's statements, checked as identities -- not as fits.

Every test here is an equality that must hold for *all* inputs, so the seeds are
irrelevant except where an expectation is estimated (and there the tolerance is
a stated number of standard errors). If one of these fails, the theorem is false,
not the tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from entrain_infer import detector, theorem


# ─────────────────────────────────────────────── statement 1: the gauge ──


def test_the_statistic_is_blind_to_a_common_mode():
    """``t(x + c 1) == t(x)``: the park cannot see its own clock."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.3, 12)
    t0 = detector.studentize(x)
    for c in (-5.0, -0.001, 0.0, 1.0, 1e3):
        shifted = detector.studentize(x + c)
        assert np.allclose(shifted, t0, rtol=0, atol=1e-12 * max(1.0, abs(c)))


def test_a_full_coalition_is_invisible_at_any_size():
    """``k = n``: everybody lying by ``b`` is ``b * 1``, a pure gauge shift."""
    for n in (2, 10, 100):
        for b in (1e-9, 1.0, 1e6):
            assert abs(detector.statistic(np.full(n, b))) < 1e-12 * max(1.0, b)


def test_the_contrast_norm_is_the_chord():
    """``||P_H 1_S||`` equals ``chord(n, k)`` for an actual indicator, any ``S``."""
    rng = np.random.default_rng(7)
    for n in (3, 10, 30):
        for k in range(n + 1):
            mask = np.zeros(n, dtype=bool)
            mask[rng.choice(n, size=k, replace=False)] = True
            norm = float(np.linalg.norm(theorem.contrast(mask.astype(float))))
            assert math.isclose(norm, theorem.chord(n, k), rel_tol=1e-12)



# ──────────────────────────────────────────────── statement 2: the chord ──


def test_exact_means_of_the_node_statistic():
    """``E t`` on both sides of the split, measured, against the closed form."""
    n, k, b, sigma = 10, 3, 2.0, 1.0
    draws = 40_000
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, sigma, (draws, n))
    x[:, :k] += b
    # The same closed form as ``detector.studentize``, applied row-wise.
    z = x / sigma
    t = (n * z - z.sum(axis=1, keepdims=True)) / math.sqrt(n * (n - 1))

    inside, outside = t[:, :k].mean(), t[:, k:].mean()
    se_in = t[:, :k].std(ddof=1) / math.sqrt(draws * k)
    se_out = t[:, k:].std(ddof=1) / math.sqrt(draws * (n - k))
    assert abs(inside - theorem.colluder_t_mean(n, k, b)) < 4 * se_in
    assert abs(outside - theorem.honest_t_mean(n, k, b)) < 4 * se_out


def test_dilution_for_a_small_group_is_the_m1_formula():
    """M1's criterion 7, derived rather than measured: ``(n-k)/(n-1)``."""
    for n in (10, 30, 100):
        for k in range(1, n // 2 + 1):
            assert math.isclose(theorem.node_gain(n, k), (n - k) / (n - 1), rel_tol=1e-12)


def test_above_half_the_flag_points_at_the_honest():
    """Deterministic, no noise: for ``k > n/2`` the argmax is an honest node."""
    n, b = 10, 1.0
    for k in range(6, n):
        x = b * (np.arange(n) < k).astype(float)
        flagged = int(np.argmax(np.abs(detector.studentize(x))))
        assert flagged >= k, f"k={k}: flagged a colluder ({flagged})"


def test_below_half_it_points_at_a_colluder():
    n, b = 10, 1.0
    for k in range(1, 5):
        x = b * (np.arange(n) < k).astype(float)
        assert int(np.argmax(np.abs(detector.studentize(x)))) < k


def test_the_b_min_ratio_is_symmetric_and_one_at_both_ends():
    n = 30
    for k in range(1, n):
        assert math.isclose(
            theorem.b_min_ratio(n, k), theorem.b_min_ratio(n, n - k), rel_tol=1e-12
        )
    assert math.isclose(theorem.b_min_ratio(n, 1), 1.0, rel_tol=1e-12)
    assert math.isclose(theorem.b_min_ratio(n, n - 1), 1.0, rel_tol=1e-12)
    assert theorem.b_min_ratio(n, n) == math.inf
    assert theorem.b_min_ratio(n, n // 2) > theorem.b_min_ratio(n, 1)


def test_the_bound_is_attained_only_at_the_ends():
    """``b_min_ratio == 1`` iff the maximum sits on a single node: ``k`` in {1, n-1}."""
    for n in (4, 10, 30):
        assert theorem.b_min_ratio(n, 1) == 1.0
        assert theorem.b_min_ratio(n, n - 1) == 1.0
        for k in range(2, n - 1):
            assert theorem.b_min_ratio(n, k) > 1.0


def test_a_coalition_at_the_bound_is_at_least_as_loud_as_one_liar():
    """The lemma behind the ceiling: ``power(k, b_min(1)·r) >= power(1, b_min(1))``.

    Same threshold, and the largest group of the coalition carries exactly the
    noncentrality of a lone liar, so raising the lie to the bound cannot make the
    coalition quieter -- and for ``2 <= k <= n-2`` it makes it louder, which is
    why the measured ``b_min(k)`` stays *below* the bound. Measured on the bench's
    own coalition model (``bench._coalition_power``), so no second implementation
    enters the argument.
    """
    from entrain_infer import bench

    alpha, target, draws, seed = 1e-3, 0.9, 4_000, 23
    for n in (10, 30):
        tau = detector.null_threshold(n, alpha)
        b1 = detector.b_min(n, 1.0, alpha, target)
        base, _ = bench._coalition_power(n, 1, b1, 1.0, tau, draws, seed)
        assert base == pytest.approx(target, abs=0.05)  # the power the bound is about
        for k in range(2, n):
            bound_b = b1 * theorem.b_min_ratio(n, k)
            at_bound, _ = bench._coalition_power(n, k, bound_b, 1.0, tau, draws, seed)
            assert at_bound >= base - 0.02, f"n={n} k={k}: bound is not a ceiling"
            if k <= n - 2:  # the max group holds m >= 2 nodes: strictly louder
                assert at_bound > base + 0.02, f"n={n} k={k}: ceiling is not strict"


def test_the_chord_ratio_is_symmetric_and_peaks_at_half():
    n = 12
    assert math.isclose(theorem.chord(n, 0), 0.0, abs_tol=1e-15)
    assert math.isclose(theorem.chord(n, n), 0.0, abs_tol=1e-15)
    best = max(range(1, n), key=lambda k: theorem.chord_ratio(n, k))
    assert best in (n // 2, n - n // 2)
    for k in range(1, n):
        assert math.isclose(
            theorem.chord_ratio(n, k), theorem.chord_ratio(n, n - k), rel_tol=1e-12
        )


def test_the_chord_test_dominates_the_node_test():
    """``chord_ratio >= node_gain`` for every ``k``, equality only at 1 and n-1."""
    for n in (4, 10, 30, 100):
        for k in range(1, n):
            assert theorem.chord_ratio(n, k) >= theorem.node_gain(n, k) - 1e-12
        assert math.isclose(
            theorem.chord_ratio(n, 1), theorem.node_gain(n, 1), rel_tol=1e-12
        )
        assert math.isclose(
            theorem.chord_ratio(n, n - 1), theorem.node_gain(n, n - 1), rel_tol=1e-12
        )
        if n > 4:
            assert theorem.chord_ratio(n, n // 2) > 1.5 * theorem.node_gain(n, n // 2)


# ────────────────────────────────── statement 3: orientation in the kernel ──


def test_two_worlds_are_one_observation():
    """``mu 1 + b 1_S`` and ``(mu+b) 1 - b 1_{S^c}`` are the same readings."""
    x_a, x_b, world_a, world_b = theorem.two_worlds(10, 4, 2.5, mu=-0.75)
    assert np.allclose(x_a, x_b, rtol=0, atol=1e-12)
    # The difference is a gauge shift: a constant vector, i.e. nothing.
    diff = x_a - x_b
    assert np.allclose(diff, np.full(10, diff.mean()), atol=1e-12)
    assert not np.any(world_a["liars"] & world_b["liars"])


def test_two_worlds_flag_the_same_node():
    """The same node, with contradictory verdicts about *why* it was flagged."""
    x_a, x_b, world_a, world_b = theorem.two_worlds(10, 4, 2.5)
    t_a, t_b = detector.studentize(x_a), detector.studentize(x_b)
    assert np.allclose(t_a, t_b, rtol=0, atol=1e-12)
    node = int(np.argmax(np.abs(t_a)))
    assert node in np.flatnonzero(world_a["liars"])
    assert node not in np.flatnonzero(world_b["liars"])
    # The two descriptions disagree about the sign as well as about the side.
    assert world_a["b"] == -world_b["b"]


def test_the_verdict_is_identical_so_no_power_can_separate_the_worlds():
    """Flagged or not, and at what risk -- the same, because ``x`` is the same."""
    x_a, x_b, _, _ = theorem.two_worlds(30, 12, 0.02)
    v_a = detector.detect(x_a, sigma=1.0, alpha=1e-3)
    v_b = detector.detect(x_b, sigma=1.0, alpha=1e-3)
    assert v_a.statistic == pytest.approx(v_b.statistic, rel=1e-12)
    assert v_a.node == v_b.node
    assert v_a.flagged == v_b.flagged


def test_a_constant_offset_is_the_gauge_of_the_estimator():
    """The frequency reading has the same kernel: ``+c`` on every node moves nothing."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.1, 8)
    assert np.allclose(detector.studentize(x + 0.37), detector.studentize(x), atol=1e-12)


# ───────────────────────────────────────────── the converse: an anchor ──


def test_the_anchor_sees_what_the_relative_test_calls_honest():
    """``k = n-1``: the park disagrees with the anchor, and only the anchor knows."""
    n, k = 10, 9
    x = (np.arange(n) < k).astype(float)  # node 9 is the honest anchor
    assert int(np.argmax(np.abs(detector.studentize(x)))) == n - 1
    # Against the anchor the same lie has noncentrality b sqrt(n-1)/sigma = 3.
    assert theorem.anchor_statistic(x, 1.0, anchor=n - 1) == pytest.approx(math.sqrt(n - 1))


def test_the_anchor_b_min_is_finite_and_much_smaller():
    """Both in absolute units: the anchored floor vs the unanchored single-node one."""
    n, sigma, alpha = 10, 1.0, 1e-3
    anchored = theorem.anchor_b_min(n, sigma, alpha, 0.9)
    unanchored = detector.b_min_analytic(n, sigma, alpha, 0.9)
    assert math.isfinite(anchored)
    # sqrt(n-1) = 3.16 times smaller, up to the two different thresholds.
    assert unanchored / anchored > 3.0
    assert theorem.anchor_power(anchored, sigma, n, alpha) == pytest.approx(0.9, abs=0.02)


def test_the_anchor_power_matches_its_definition():
    n, sigma, alpha = 20, 2.0, 1e-3
    for k in (1, 5, n - 1):
        b = theorem.anchor_b_min(n, sigma, alpha, 0.9, k=k)
        assert theorem.anchor_power(b, sigma, n, alpha, k=k) == pytest.approx(0.9, abs=1e-3)
    assert theorem.anchor_power(0.0, sigma, n, alpha) == pytest.approx(alpha, abs=1e-12)


def test_the_anchor_is_a_trade_not_a_free_lunch():
    """Worse than the plain GLRT against one liar, far better against all but one."""
    alpha = 1e-3
    for n in (10, 30, 100):
        relative = lambda k: detector.b_min_analytic(n, 1.0, alpha) * theorem.b_min_ratio(n, k)
        assert theorem.anchor_b_min(n, 1.0, alpha, 0.9, k=1) > relative(1)
        assert theorem.anchor_b_min(n, 1.0, alpha, 0.9, k=n - 1) < relative(n - 1)
        # ... and the crossover sits where the module says it does.
        star = theorem.anchor_crossover(n)
        assert theorem.anchor_b_min(n, 1.0, alpha, 0.9, k=star) < relative(star)
        if star > 1:
            assert theorem.anchor_b_min(n, 1.0, alpha, 0.9, k=star - 1) > relative(star - 1)
        assert abs(star - math.sqrt(n)) <= 2.0  # the same order as sqrt(n)


def test_an_anchor_that_lies_is_the_kernel_again():
    """If the reference itself is the liar, ``y`` is zero: undetectable, as before."""
    n, b = 10, 5.0
    x = np.full(n, b)  # everybody, the anchor included, is off by b
    assert abs(theorem.anchor_statistic(x, 1.0, anchor=0)) < 1e-12
    assert theorem.anchor_b_min(n, 1.0, 1e-3, 0.9, k=0) == math.inf


# ──────────────────────────────────────────────────────── the chord test ──


def test_the_chord_statistic_is_standard_normal_under_h0():
    n, k, sigma, draws = 12, 3, 1.0, 40_000
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, sigma, (draws, n))
    liars = np.arange(n) < k
    z = np.array([theorem.chord_statistic(row, sigma, liars) for row in x])
    assert abs(z.mean()) < 4 * z.std(ddof=1) / math.sqrt(draws)
    assert math.isclose(z.std(ddof=1), 1.0, rel_tol=0.02)


def test_the_chord_statistic_has_the_chord_as_its_mean_shift():
    n, k, b, sigma = 12, 4, 1.5, 1.0
    x = b * (np.arange(n) < k).astype(float)
    liars = np.arange(n) < k
    assert math.isclose(
        theorem.chord_statistic(x, sigma, liars),
        b * theorem.chord(n, k) / sigma,
        rel_tol=1e-12,
    )


def test_for_a_full_coalition_the_chord_test_divides_by_zero():
    assert theorem.chord_b_min(10, 10, 1.0, 1e-3) == math.inf
    assert theorem.chord_b_min(10, 0, 1.0, 1e-3) == math.inf


def test_chord_power_hits_its_target_at_chord_b_min():
    n, k, sigma, alpha = 16, 5, 3.0, 1e-3
    b = theorem.chord_b_min(n, k, sigma, alpha, 0.9)
    assert theorem.chord_power(b, n, k, sigma, alpha) == pytest.approx(0.9, abs=1e-3)


def test_the_bench_coalition_model_is_the_detector_model():
    """``bench._coalition_rows`` reproduces ``detector`` row by row, node included.

    A bench that grows its own statistic can prove anything; this pins the
    vectorised coalition model to the detector the report quotes.
    """
    from entrain_infer import bench

    n, k, b, sigma, draws = 10, 4, 0.01, 0.002, 60
    stats, nodes = bench._coalition_rows(n, k, b, sigma, draws, seed=3)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((draws, n)) * sigma
    x[:, :k] += b
    for i in range(draws):
        assert stats[i] == pytest.approx(detector.statistic(x[i] / sigma), rel=1e-12)
        assert int(nodes[i]) == int(np.argmax(np.abs(detector.studentize(x[i] / sigma))))


# ───────────────────────────────────────────────────── prediction table ──


def test_every_prediction_row_is_internally_consistent():
    n = 10
    rows = theorem.predicted_rows([n], {n: range(1, n + 1)})
    assert [row["k"] for row in rows] == list(range(1, n + 1))
    by_k = {row["k"]: row for row in rows}
    assert by_k[1]["b_min_ratio"] == pytest.approx(1.0)
    assert by_k[n]["b_min_ratio"] == math.inf
    for k in range(1, n):
        assert by_k[k]["b_min_ratio"] == pytest.approx(by_k[n - k]["b_min_ratio"])
        expected = "colluder" if k < n - k else ("honest" if k > n - k else "tie")
        assert by_k[k]["flagged_side"] == expected
    assert by_k[1]["flagged_side"] == "colluder"
    assert by_k[n - 1]["flagged_side"] == "honest"
    # At k = n there is no second side, so nobody is named by a silent statistic.
    assert by_k[n]["flagged_side"] == "none"

