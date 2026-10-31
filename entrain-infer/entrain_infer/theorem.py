"""M2: the collusion theorem -- a no-go statement, and its only converse.

The observation
---------------
A park of ``n`` producers is read as ``x`` (``model.Observation``). The GLRT of
``detector`` works on the studentized contrasts

    t_i = (n x_i - sum_j x_j) / sqrt(n (n-1)),

and ``t(x + c * 1) = t(x)`` for every ``c``: adding the same number to every
producer moves nothing. That is not a numerical accident. It is the *only*
symmetry the park has, and everything below follows from it.

Statement 1 (gauge)
-------------------
The observation lives in the quotient ``R^n / span{1}`` -- the space of
contrasts, of dimension ``n-1``. A lie ``b`` carried by a set ``S`` enters the
observation only through its contrast ``b * P_H 1_S``. Hence

    |S| = n  (everybody lies by the same b)  =>  P_H 1_S = 0  =>  invisible

at any ``b``, to any test that respects the gauge, at any power. This is the
same symmetry that makes a constant timestamp offset invisible to a frequency
reading (report §8): there, too, the lie is a common-mode shift.

Statement 2 (chord)
-------------------
The length of that contrast is the chord between the two clusters::

    c(S) = ||P_H 1_S|| = sqrt(k (n-k) / n),        k = |S|,

so a split park is visible exactly to the extent that it is *split*: zero at
``k = 0`` and ``k = n``, maximal at ``k = n/2``. The per-node statistic of M1
sees the same fact through its own lens. Under ``x = mu 1 + b 1_S`` exactly

    E t_i = + b (n-k) / sqrt(n (n-1))     for i in S,
    E t_i = - b  k    / sqrt(n (n-1))     for i not in S,

which is M1's dilution ``(n-k)/(n-1)`` while ``k <= n/2`` -- and *inversion*
once ``k > n/2``, where the largest statistics belong to the honest minority:
the detector names the honest. Nothing is broken here; the statistic reports
the only thing it can see, the chord, and a chord does not know which end is
which.

Statement 3 (orientation is in the kernel)
------------------------------------------
For every ``S``, ``b`` and ``mu``::

    mu * 1 + b * 1_S  ==  (mu + b) * 1 + (-b) * 1_{S^c}

identically, not approximately -- as observations. Two worlds, "k nodes lie by
+b" and "n-k nodes lie by -b while the park clock is off by +b", produce the
same ``x``, hence the same ``t``, the same flagged node and the same ``b_min``.
No test of this observation can tell them apart, because their difference is a
gauge transformation: the orientation of the split is not hard to estimate, it
is not in the data. Only the chord is.

Corollary (the only converse: an anchor)
----------------------------------------
Trust cannot be derived from the park; it can only be pinned. Declare one
node's clock true and the quotient collapses: ``y_i = x_i - x_a`` has a *known*
mean under H0, so the question stops being "which node deviates" (relative,
``n-1`` contrasts) and becomes "does the remainder deviate from the anchor"
(absolute). With ``k`` liars past the anchor the SNR is

    delta = b k / sqrt(n-1),        b_min = (tau + z_{1-beta}) sigma sqrt(n-1) / k,

finite for every ``k`` -- including ``k = n-1``, where the relative test is as
insensitive as against a lone liar *and* names the honest node, while the anchor
is at its best (``b_min = (tau+z) sigma / sqrt(n-1)``).

And the anchor is a trade, not a free lunch: its floor grows linearly in ``k``
while the relative one grows in ``n-k``, so it *loses* to the plain GLRT below
``k ~ sqrt(n)`` and wins above (``anchor_crossover``). One assumption, bought at
the price of precision against the few, pays for the many.

Lemma (why the bound on ``b_min`` is one-sided)
-----------------------------------------------
Put ``m = max(k, n-k)``, ``s = min(k, n-k)`` and ``r = (n-1)/m`` -- the number
``b_min_ratio`` returns. Raise the coalition's lie to ``b' = b r`` and, taking
``k <= n/2`` without loss (the map is symmetric), its ``m`` largest statistics sit
at exactly the noncentrality a lone liar has at ``b``::

    E t_i = b' (n-k) / sqrt(n (n-1)) = b sqrt((n-1)/n),   i in S,

while the ``s`` nodes on the other side sit at ``-b' s / sqrt(n (n-1)) ``. The
threshold ``tau`` is the same in both worlds, so for the target power of one liar

    1 - (1-p)^m  <=  power(coalition at b')  <=>  b_min(k) <= b_min(1) r,

where ``p`` is the chance that a single node at that noncentrality crosses
``tau``. Equality needs ``m = 1`` (``k = n-1``, or ``k = 1`` where ``r = 1`` and
the configuration is the same one): there one statistic carries the maximum in
both worlds. For ``2 <= k <= n-2`` the bound is *strict*, because
``(1-p)^(m-1) < (1-alpha)^(n-1)``, so ``m`` nodes sharing the maximal
noncentrality fire more often than the one node that alone sets ``b_min(1)``.
This is why the bench reports the slack as a column instead of hiding it behind a
tolerance: the bound is a ceiling, not an equality, and the ceiling is reached
only at the two ends.

What is *not* claimed
---------------------
Nothing here says the GLRT is the best test of a coalition. It is the test of
*one* liar; a coalition is better tested by the chord itself, and
``chord_b_min`` is that test for a coalition known in advance (an operator's own
devices, say). The theorem says the chord is all there is -- any test is a
function of the same quotient -- so no test can beat the chord's information,
though one *ordered by the chord* beats one ordered by the node maximum. The
report measures both, side by side.

Everything in this module is closed form: no Monte Carlo, no seed, no
thresholds. Numbers here are proved, or they are not here.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


# ─────────────────────────────────────────────────────────────── contrasts ──


def contrast(v: np.ndarray) -> np.ndarray:
    """``P_H v = v - mean(v)``: the part of ``v`` that the park can see."""
    v = np.asarray(v, dtype=float).ravel()
    return v - v.mean()


def chord(n: int, k: int) -> float:
    """``||P_H 1_S|| = sqrt(k (n-k) / n)`` -- the only observable of a split.

    Zero when nobody or everybody lies, maximal at ``k = n/2``. It is the length
    of the chord between the two clusters in contrast space, and by Statement 2
    it is what *any* test of this observation is a function of.
    """
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    return math.sqrt(k * (n - k) / n)


def chord_ratio(n: int, k: int) -> float:
    """``chord(n, k) / chord(n, 1)`` -- detectability relative to one liar.

    Depends only on ``min(k, n-k)``: the same split seen from either side. That
    is the chord symmetry in one number.
    """
    base = chord(n, 1)
    if base == 0.0:
        raise ValueError("need n >= 2")
    return chord(n, k) / base


def colluder_t_mean(n: int, k: int, b: float) -> float:
    """``E t_i`` for ``i`` inside a coalition of size ``k`` lying by ``b``."""
    return b * (n - k) / math.sqrt(n * (n - 1))


def honest_t_mean(n: int, k: int, b: float) -> float:
    """``E t_i`` for ``i`` outside it -- negative, larger in size for ``k > n/2``."""
    return -b * k / math.sqrt(n * (n - 1))


def b_eff_factor(n: int, k: int) -> float:
    """What ``bench.probe`` measures on the colluders, in units of ``b``: ``(n-k)/n``.

    ``probe`` reports ``mean(t[nodes]) * sigma * sqrt((n-1)/n)``, which is
    ``b (n-k)/n`` under the coalition model -- the number M1 checked against
    ``(n-k)/(n-1)`` relative to a single liar.
    """
    return (n - k) / n


def node_gain(n: int, k: int) -> float:
    """Detectability through the *single-node* statistic, relative to one liar.

    ``max(k, n-k) / (n-1)``: the noncentrality of the largest ``|t|`` divided by
    that of a single liar. Equals 1 at ``k = 1`` and at ``k = n-1`` -- a
    coalition of all but one is exactly as detectable as one liar, its flag just
    points the other way -- and 0 at ``k = n``, where there is no second side to
    compare against and every ``t`` is identically zero.
    """
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n")
    if k == n:
        return 0.0
    return max(k, n - k) / (n - 1)


def b_min_ratio(n: int, k: int) -> float:
    """``b_min(k) / b_min(1)`` for the single-node test: ``(n-1)/max(k, n-k)``.

    Precisely: this is the exact ratio of *noncentralities* (``node_gain``), which
    makes it an **upper bound** on the true ``b_min`` ratio and an exact value at
    the two ends. The bound is one-sided because a coalition puts
    ``max(k, n-k)`` statistics at the maximal noncentrality where a single liar
    puts one, and a maximum over ``m`` normals sharing that noncentrality is
    larger than one of them -- so a coalition is *easier* to catch than the
    noncentrality alone suggests, never harder (the lemma in the module docstring
    carries the argument). The bound is attained exactly at ``k = 1`` and
    ``k = n-1`` (a single statistic carries the maximum), and the gap peaks at
    ``k ~ n/2``, where about ``n/2`` nodes share it. The bench measures the gap
    instead of hiding it behind a tolerance; ``tests/test_theorem.py`` checks both
    the equality at the ends and the direction of the inequality.

    ``inf`` at ``k = n``: no offset is detectable, because there is nothing to
    compare against (Statement 1). The map is symmetric about ``n/2``, and it is
    never below 1: collusion never makes a lie easier to see *per node*.
    """
    gain = node_gain(n, k)
    return math.inf if gain == 0.0 else 1.0 / gain


def flagged_side(n: int, k: int) -> str:
    """Whose node the single-node test points at: the liars, or the honest.

    ``"colluder"`` for ``k < n/2``, ``"honest"`` for ``k > n/2``, ``"tie"`` at an
    exact half (broken by noise, so the label is a coin there), and ``"none"`` at
    ``k = n``, where there is no second side and the statistic is silent: nobody
    is named because nothing is left to name.
    """
    if k == n:
        return "none"
    if n - k > k:
        return "colluder"
    if k > n - k:
        return "honest"
    return "tie"


def predicted_rows(ns, ks_by_n=None) -> list[dict[str, object]]:
    """The theorem's predictions, for every ``n`` and every coalition size.

    Each row states, before any measurement: how far the flag is diluted
    (``b_min_ratio``), whose node it lands on (``flagged_side``), how much of the
    lie the statistic keeps (``b_eff_factor``), and what the chord does
    (``chord_ratio``, symmetric, maximal at half). A bench that disagrees with a
    row anywhere is a refutation, not a tuning problem.
    """
    rows: list[dict[str, object]] = []
    for n in ns:
        ks = ks_by_n.get(n, range(1, n + 1)) if ks_by_n else range(1, n + 1)
        for k in ks:
            rows.append(
                {
                    "n": int(n),
                    "k": int(k),
                    "chord_ratio": chord_ratio(n, k),
                    "b_eff_factor": b_eff_factor(n, k),
                    "node_gain": node_gain(n, k),
                    "b_min_ratio": b_min_ratio(n, k),
                    "flagged_side": flagged_side(n, k),
                }
            )
    return rows


# ─────────────────────────────────────────────────────────────── two worlds ──


def two_worlds(n: int, k: int, b: float, mu: float = 0.0):
    """The same observation, two contradictory descriptions of who lied.

    World A: nodes ``0..k-1`` lie by ``+b``; the park clock reads ``mu``.
    World B: the other ``n-k`` nodes lie by ``-b``; the park clock reads
    ``mu + b``.

    Returns ``(x_a, x_b, world_a, world_b)``; each world is a dict with its lying
    mask, its sign and its park mean. ``x`` is built twice, once from each
    description, so that a caller can assert the two constructions agree.
    """
    if not 0 < k < n:
        raise ValueError("both worlds must be non-empty: 0 < k < n")
    liars_a = np.zeros(n, dtype=bool)
    liars_a[:k] = True
    x_a = mu + b * liars_a.astype(float)
    x_b = (mu + b) + (-b) * (~liars_a).astype(float)
    world_a = {"liars": liars_a, "b": b, "mu": mu}
    world_b = {"liars": ~liars_a, "b": -b, "mu": mu + b}
    return x_a, x_b, world_a, world_b


# ────────────────────────────────────────────────────────────── the anchor ──


def anchor_reading(x: np.ndarray, anchor: int = 0) -> np.ndarray:
    """``x_i - x_anchor`` for every ``i != anchor``: the gauge, fixed by hand.

    Returns ``n-1`` numbers whose mean under H0 is *known* to be zero -- which is
    exactly the information the relative test throws away.
    """
    x = np.asarray(x, dtype=float).ravel()
    if not 0 <= anchor < x.size:
        raise ValueError("anchor out of range")
    return np.delete(x - x[anchor], anchor)


def anchor_statistic(x: np.ndarray, sigma: float, anchor: int = 0) -> float:
    """``sqrt(m) * mean(y) / sigma`` ~ ``N(delta, 1)``, ``m = n-1``, ``y = x - x_a``.

    Tests "the rest of the park agrees with the anchor". Under H0 this is exactly
    standard normal -- no null table, no ``n``-dependent quantile, because the
    mean under H0 is known rather than profiled.

    With ``k`` liars of the *same sign* among the other ``m`` nodes (a coalition,
    by the model of this module: one ``b`` for the whole set) the mean shift is
    ``delta = b k / sqrt(n-1)``: linear in ``k``, which is why an anchor is a
    trade and not a free lunch (``anchor_crossover``). Opposite signs would
    cancel, and this test would see nothing -- which is why the theorem is stated
    for a single ``b`` and not for arbitrary deviations.
    """
    y = anchor_reading(x, anchor)
    return float(math.sqrt(y.size) * y.mean() / sigma)


def anchor_noncentrality(n: int, k: int, b: float) -> float:
    """``b k / sqrt(n-1)``: the anchored test's SNR for ``k`` liars past the anchor."""
    return b * k / math.sqrt(n - 1)


def anchor_b_min(
    n: int, sigma: float, alpha: float, target_power: float = 0.9, k: int | None = None
) -> float:
    """``(tau + z_{1-beta}) sigma sqrt(n-1) / k``: the anchored floor for ``k`` liars.

    ``k=None`` means the anti-coalition ``k = n-1``, where the anchored floor is
    the best in the whole module: ``(tau+z) sigma / sqrt(n-1)``. ``k = 0`` has no
    floor (there is nothing to detect) and ``k = n`` has none either -- at ``k = n``
    the anchor itself lies, ``y`` is zero, and the gauge is back. Both return
    ``inf``.

    Note the shape: the anchored ``b_min`` *decreases* with ``k`` while the
    relative one *increases* up to ``n/2``. That is the whole trade -- an anchor
    converts a relative comparison into an absolute one, which costs the many
    honest against one liar and buys the honest against many.
    """
    if k is None:
        k = n - 1
    if k <= 0 or k >= n:
        return math.inf
    tau = NormalDist().inv_cdf(1.0 - alpha)
    z = NormalDist().inv_cdf(target_power)
    return float((tau + z) * sigma * math.sqrt(n - 1) / k)


def anchor_power(
    b: float, sigma: float, n: int, alpha: float, k: int | None = None
) -> float:
    """Power of the anchored test, exactly: ``1 - Phi(tau - delta)``."""
    if k is None:
        k = n - 1
    tau = NormalDist().inv_cdf(1.0 - alpha)
    delta = anchor_noncentrality(n, k, abs(b)) / sigma
    return float(1.0 - NormalDist().cdf(tau - delta))


def anchor_crossover(n: int, alpha: float = 1e-3, target_power: float = 0.9) -> int:
    """Smallest ``k`` at which the anchored test beats the relative one.

    Two effects, one direction. The anchored floor grows in ``k`` where the
    relative one grows in ``n-k``, *and* the anchored test needs no multiplicity
    correction (one statistic, not ``n``). Comparing the floors with the closed
    form thresholds -- ``Phi^{-1}(1-alpha)`` against a Bonferroni
    ``Phi^{-1}(1-alpha/(2n))`` -- gives

        k* = n (tau_a + z) / ((tau_b + z) sqrt(n) + (tau_a + z)),

    which is about ``sqrt(n)`` nodes for the risks the bench declares. Below it
    the anchor is *worse* than the plain GLRT (a lone liar is compared to one
    reference instead of to ``n-1`` honest nodes); above it the anchor wins, and
    at ``k = n-1`` the relative test names the honest node while the anchor is at
    its best. Governance reading: an anchor is worth its assumption only against
    coalitions bigger than ``sqrt(n)``.

    The number is conservative by construction: the Bonferroni threshold is
    above the Monte-Carlo one used in ``detector``, so the true crossover moves
    down by at most one node.
    """
    tau_a = NormalDist().inv_cdf(1.0 - alpha)
    tau_b = NormalDist().inv_cdf(1.0 - alpha / (2.0 * n))
    z = NormalDist().inv_cdf(target_power)
    num = n * (tau_a + z)
    den = (tau_b + z) * math.sqrt(n) + (tau_a + z)
    return int(math.floor(num / den)) + 1





# ──────────────────────────────────────────────── the chord test (known S) ──


def chord_statistic(x: np.ndarray, sigma: float, liars: np.ndarray) -> float:
    """``z = (mean_S - mean_rest) * chord(n,k) / sigma`` ~ ``N(b chord / sigma, 1)``.

    The efficient test when the coalition ``S`` is known in advance (one
    operator's own devices): the two-cluster contrast, standardised by its own
    variance. Its noncentrality is the chord, not the node maximum, so at
    ``k = n/2`` it beats ``max_k |t_k|`` -- measured in the report.
    """
    x = np.asarray(x, dtype=float).ravel()
    liars = np.asarray(liars, dtype=bool).ravel()
    n = x.size
    k = int(liars.sum())
    if liars.size != n or not 0 < k < n:
        raise ValueError("liars must be a non-empty proper subset")
    diff = x[liars].mean() - x[~liars].mean()
    var = sigma * sigma * (1.0 / k + 1.0 / (n - k))
    return float(diff / math.sqrt(var))


def chord_b_min(
    n: int, k: int, sigma: float, alpha: float, target_power: float = 0.9
) -> float:
    """``(tau + z_{1-beta}) sigma / chord(n,k)``, ``tau = Phi^{-1}(1-alpha/2)``.

    Two-sided because a coalition may lie in either direction. ``inf`` at
    ``k = n``: the chord is zero, so the test has nothing to divide by -- which is
    Statement 1 again, now as a division by zero rather than as a silence.
    """
    c = chord(n, k)
    if c == 0.0:
        return math.inf
    tau = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z = NormalDist().inv_cdf(target_power)
    return float((tau + z) * sigma / c)


def chord_power(b: float, n: int, k: int, sigma: float, alpha: float) -> float:
    """Power of the chord test at a known coalition of size ``k``."""
    c = chord(n, k)
    tau = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    delta = abs(b) * c / sigma
    nd = NormalDist()
    return float(nd.cdf(delta - tau) + nd.cdf(-delta - tau))
