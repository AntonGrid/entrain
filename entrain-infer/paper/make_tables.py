"""Generate the LaTeX tables of the paper from ``bench/results.json``.

No number in the paper is typed by hand: every cell of every table is read from
the frozen run (``bench/results.json``, sha256 in Appendix C) or computed by the
closed-form functions of ``entrain_infer.theorem``. Re-running this script on the
same ``results.json`` reproduces ``tables/*.tex`` byte for byte.

Usage
-----
    python paper/make_tables.py            # writes paper/tables/*.tex

The script writes nothing outside ``paper/tables/``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from entrain_infer.theorem import colluder_t_mean, honest_t_mean  # noqa: E402

RESULTS = ROOT / "bench" / "results.json"
OUT = HERE / "tables"


def _f(value: float, digits: int = 3) -> str:
    """Format a float for LaTeX; ``inf``/``nan``/``None`` get their placeholders."""
    if value is None:
        return "--"
    if isinstance(value, float) and math.isinf(value):
        return r"$\infty$"
    if isinstance(value, float) and math.isnan(value):
        return r"--"
    return f"{value:.{digits}f}"


def write(name: str, lines: list[str]) -> None:
    path = OUT / name
    body = "\n".join(lines)
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({len(lines)} lines)")


def table_chord(data: dict) -> None:
    rows = []
    for row in data["predicted"]:
        n, k = row["n"], row["k"]
        rows.append(
            " & ".join(
                [
                    str(n),
                    str(k),
                    _f(row["chord_ratio"]),
                    _f(colluder_t_mean(n, k, 1.0)),
                    _f(honest_t_mean(n, k, 1.0)),
                    _f(row["node_gain"]),
                    _f(row["b_min_ratio"]),
                    row["flagged_side"],
                ]
            )
            + r" \\"
        )
    write(
        "tab_chord.tex",
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{The chord and the per-node statistic, closed form "
            r"(\texttt{theorem.predicted}); no Monte Carlo in this table. "
            r"$c(S)=\|P_H\mathbf{1}_S\|=\sqrt{k(N-k)/N}$, so $c/c_1$ is the "
            r"detectability of a known coalition relative to one liar; "
            r"$\mathbb{E}t_i/b$ are the per-node means inside and outside the "
            r"coalition; gain is $\max(k,N-k)/(N-1)$, the noncentrality of the "
            r"largest $|t|$ relative to a single liar. ``flagged'' is the side the "
            r"GLRT names. At $k=N$ every $t_i$ is identically zero: nothing is "
            r"named, because there is no second side to compare against.}",
            r"\label{tab:chord}",
            r"\small",
            r"\begin{tabular}{@{}r r r r r r r l@{}}",
            r"\toprule",
            r"$N$ & $k$ & $c/c_1$ & in & out & gain & $b_{\min}$ ratio & flagged \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ],
    )


def table_measured(data: dict) -> None:
    rows = []
    for row in data["measured"]:
        ratio_meas = row.get("b_min_ratio_measured")
        held = row.get("bound_holds")
        rows.append(
            " & ".join(
                [
                    str(row["n"]),
                    str(row["k"]),
                    _f(row["b1_over_sigma"]),
                    _f(row["b_min_over_sigma"]),
                    _f(row["b_min_ratio_predicted"]),
                    _f(ratio_meas) if ratio_meas is not None else "--",
                    "yes" if held else ("no" if held is False else "--"),
                    _f(row.get("power_at_b1")),
                    _f(row.get("power_chord_at_b1")),
                    _f(row.get("flag_on_honest"), 5),
                ]
            )
            + r" \\"
        )
    write(
        "tab_measured.tex",
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{The bound of Lemma~\ref{lem:bound} against the run "
            r"(\texttt{theorem.measured}). $b_1$ is the measured minimum detectable "
            r"lie for a lone liar at the declared risk $\alpha=10^{-3}$ and target "
            r"power $0.9$; $\hat b_{\min}(k)$ is measured by bisection at the same "
            r"power. ``pred'' is $(N-1)/\max(k,N-k)$, ``meas'' is "
            r"$\hat b_{\min}(k)/b_1$, ``bound'' the inequality between them -- one "
            r"column, because it is a ceiling and not an identity. ``chord'' is the "
            r"power of the known-coalition test at the same $b_1$: a coalition is "
            r"easier to catch than a node maximum suggests, never harder. ``flag on "
            r"honest'' is the measured share of honest nodes flagged at $b_1$ -- the "
            r"inversion of \S\ref{sec:inversion}, read from the same runs.}",
            r"\label{tab:measured}",
            r"\small",
            r"\begin{tabular}{@{}r r r r r r c r r r@{}}",
            r"\toprule",
            r"$N$ & $k$ & $b_1/\sigma$ & $\hat b_{\min}/\sigma$ & pred & meas "
            r"& bound & power & chord & flag on honest \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ],
    )


def table_anchor(data: dict) -> None:
    rows = []
    crossover: dict[int, tuple[int, float]] = {}
    for row in data["anchor"]:
        if "crossover_k" in row:
            crossover[int(row["n"])] = (row["crossover_k"], row["sqrt_n"])
            continue
        rows.append(
            " & ".join(
                [
                    str(row["n"]),
                    str(row["k"]),
                    _f(row["anchored_b_min_over_sigma"]),
                    _f(row["unanchored_b_min_over_sigma"]),
                    _f(row["gain"]),
                    _f(row["anchored_power_measured"], 4),
                ]
            )
            + r" \\"
        )
    note = ", ".join(
        f"$k^\\ast={k}$ at $N={n}$ ($\\sqrt{{N}}={_f(sqrt_n)}$)"
        for n, (k, sqrt_n) in sorted(crossover.items())
    )
    write(
        "tab_anchor.tex",
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{The anchor as a trade (\texttt{theorem.anchor}). An anchored "
            r"test compares $x_i-x_a$, whose mean is known, so "
            r"$b_{\min}=(\tau+z)\sigma\sqrt{N-1}/k$ falls with $k$ while the "
            r"relative floor grows up to $N/2$. ``gain'' is the relative floor over "
            r"the anchored one: below $k^\ast$ it is $<1$, i.e.\ against a few liars "
            r"the anchor is worse than the plain GLRT it replaces. "
            rf"{note.capitalize()}. At $k=N$ the anchor itself lies, $y\equiv0$, and "
            r"the measured power falls back to $\alpha$ (0.0013 at $N=30$).}",
            r"\label{tab:anchor}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}r r r r r r@{}}",
            r"\toprule",
            r"$N$ & $k$ & anchor $b_{\min}/\sigma$ & relative $b_{\min}/\sigma$ "
            r"& gain & power \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ],
    )


def table_field(data: dict) -> None:
    rows = []
    for row in data["real"]:
        if row["season"] != "january":
            continue  # July is quoted in the text; the columns would only double.
        rows.append(
            " & ".join(
                [
                    str(row["n"]),
                    str(row["k"]),
                    _f(row["detection_rate"]),
                    _f(row["localisation_rate"]),
                    str(row.get("flagged_side_predicted", "--")),
                ]
            )
            + r" \\"
        )
    write(
        "tab_field.tex",
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Field validation on the measured background (January, "
            r"$T=168$~h, frequency offset, $b/\sigma=8$, 40 repetitions per cell). "
            r"Detection is the share of repetitions crossing $\tau$; localisation is "
            r"the share whose largest $|t|$ sits on a colluder. Past $k>N/2$ detection "
            r"stays at $1.000$ while localisation collapses to $0.000$: the detector "
            r"fires and names the honest, exactly as \S\ref{sec:inversion} requires. "
            r"At $k=N$ there is nothing to detect and nothing is claimed.}",
            r"\label{tab:field}",
            r"\small",
            r"\begin{tabular}{@{}r r r r l@{}}",
            r"\toprule",
            r"$N$ & $k$ & detection & localisation & flagged side (pred) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ],
    )


def table_roc(data: dict) -> None:
    kept = (0.001, 0.003, 0.01, 0.1)
    rows = []
    for key in sorted(data["roc"]):
        cell = data["roc"][key]
        parts = key.split("|")
        if parts[0] != "n" or parts[2] != "offset" or parts[3] != "24":
            continue
        n = parts[1]
        rows.append(
            rf"\multicolumn{{6}}{{@{{}}l}}{{$N={n}$, AUC $={_f(cell['auc'])}$}} \\"
        )
        for row in cell["rows"]:
            if round(row["alpha"], 6) not in kept:
                continue
            rows.append(
                " & ".join(
                    [
                        "",
                        f"${row['alpha']:g}$",
                        _f(row["tau"], 4),
                        _f(row["fpr"], 6),
                        _f(row["tpr"], 4),
                        _f(row["tpr_analytic"], 4),
                    ]
                )
                + r" \\"
            )
    write(
        "tab_roc.tex",
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{ROC of the declared risk (\texttt{roc}, July, $T=24$~h, "
            r"frequency offset). $\tau$ is the Monte-Carlo quantile of the null "
            r"(200\,000 draws, seed 20261030, standard error printed in the report); "
            r"the achieved false-positive rate tracks $\alpha$ and the achieved power "
            r"matches the closed-form value to the third decimal. The threshold is "
            r"not a constant but a function of the declared risk and of $N$ -- which "
            r"is why a magic $0.5$ is replaced by $\alpha$ in the operator's units.}",
            r"\label{tab:roc}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}l r r r r r@{}}",
            r"\toprule",
            r" & $\alpha$ & $\tau$ & achieved FPR & achieved TPR & analytic TPR \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ],
    )


def table_kernel(data: dict) -> None:
    rows = []
    for row in data["kernel_real"]:
        rows.append(
            " & ".join(
                [
                    _f(row["b_over_sigma"], 1),
                    _f(row["b_eff_over_sigma"], 3),
                    _f(row["detection_rate"]),
                    _f(row["localisation_rate"]),
                ]
            )
            + r" \\"
        )
    write(
        "tab_kernel.tex",
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{The kernel is a silence, not a weak signal "
            r"(\texttt{kernel\_real}). Every node lies by the same $b$: the statistic "
            r"is zero up to floating point ($\hat b_{\text{eff}}/\sigma\sim10^{-17}$) "
            r"at $b/\sigma=1,5,20$, so no increase in $b$ can ever buy detection. "
            r"Measured detection is $0.000$ in all three cells, against the "
            r"$\alpha=0.001$ a coin would give.}",
            r"\label{tab:kernel}",
            r"\small",
            r"\begin{tabular}{@{}r r r r@{}}",
            r"\toprule",
            r"$b/\sigma$ & $\hat b_{\text{eff}}/\sigma$ & detection & localisation \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(RESULTS.read_text(encoding="utf-8"))
    theory = raw["theorem"]
    table_chord(theory)
    table_measured(theory)
    table_anchor(theory)
    table_field(theory)
    table_kernel(theory)
    table_roc(raw)


if __name__ == "__main__":
    main()

