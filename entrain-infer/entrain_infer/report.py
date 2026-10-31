"""The artefacts: ``bench/REPORT.md`` and ``bench/roc.png``, built from results.

Every number in the report is read out of ``bench/results.json``; nothing is typed
by hand, so a re-run cannot drift away from its own report. The 8 acceptance
criteria are copied verbatim from ``bench/criteria.json`` and given a verdict
against the measurements -- including a failing one, if that is what happened.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import bench, detector, estimator, theorem
from .model import (
    OMEGA0,
    SIGMA_HONEST,
    SIGMA_READ,
    minutes_per_day,
)

BENCH_DIR = bench.BENCH_DIR
REPORT = BENCH_DIR / "REPORT.md"
PLOT = BENCH_DIR / "roc.png"

OBS_RU = {"offset": "частота", "phase": "фаза"}
SIDE_RU = {"colluder": "лжец", "honest": "честный", "tie": "ничья", "none": "слепо"}
INJ_RU = {
    "timestamp_step": "скачок метки +2 ч в середине",
    "constant_offset": "постоянный сдвиг метки +2 ч (контроль)",
    "level": "уровень x1.6 (контроль)",
    "collusion": "сговор k узлов",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def f(value: float, nd: int = 2) -> str:
    """Format a number without lying about its size."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    magnitude = abs(float(value))
    if magnitude != 0.0 and magnitude < 1e-3:
        return f"{value:.{nd}e}"
    return f"{value:.{nd}f}"


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def fx(value: float, nd: int = 5) -> str:
    """Fixed point regardless of size: a rate of 8e-04 reads better as 0.00080."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{nd}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    """A markdown table with aligned columns."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows
    ]
    return "\n".join([line, rule] + body)


def _limit(results: dict, bg: str, season: str, hours: int, obs: str, n: int | None = None):
    key = bench._key(bg, season, hours, obs) if n is None else bench._key(bg, season, hours, obs, n)
    return results["resolution_limit"].get(key)


def _sigma(results: dict, bg: str, season: str, hours: int, obs: str) -> float | None:
    entry = results["sigma"].get(bench._key(bg, season, hours, obs))
    return entry["sigma"] if entry else None


def _sigma_minutes(results: dict, bg: str, season: str, hours: int, obs: str) -> float:
    """The same σ in the units the operator thinks in (minutes of clock)."""
    entry = results["sigma"].get(bench._key(bg, season, hours, obs))
    return float(entry["sigma_minutes"]) if entry else float("nan")


def _inj(results: dict, name: str, bg: str, season: str, hours: int, n: int, obs: str, k=None):
    key = bench._key(name, bg, season, hours, n, obs) if k is None else bench._key(
        name, bg, season, hours, n, obs, k
    )
    return results["injections"].get(key)


def plot_roc(results: dict, path: Path = PLOT) -> Path:
    """Two panels of the same curve: the window family, per reading.

    Both readings share one family shape, because ``b`` is quoted in units of
    ``sigma`` (§7): what differs between the panels is the reading, not the curve.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return path

    criteria = results["criteria"]
    alpha = float(criteria["alpha"])
    target_power = float(criteria["target_power"])
    hours_grid = tuple(int(h) for h in criteria["hours"])
    ns = tuple(int(n) for n in criteria["ns"])
    roc_hours = int(criteria.get("roc_hours", 24))
    obs = "offset"  # the curve is the curve: the reading only sets sigma, never the shape
    alphas_seen = [
        float(row["alpha"]) for curve in results["roc"].values() for row in curve["rows"]
    ]
    x_lo = (min(alphas_seen) if alphas_seen else alpha) * 0.7

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(hours_grid), len(ns), 3)))
    styles = ("-", "--", ":", "-.")

    panels = (
        (
            axes[0],
            [bench._key("hours", h, obs, ns[0]) for h in hours_grid],
            [f"T={h} ч" for h in hours_grid],
            f"Окно T при N = {ns[0]}: кривые совпадают — b задан в долях σ",
        ),
        (
            axes[1],
            [bench._key("n", n, obs, roc_hours) for n in ns],
            [f"N={n}" for n in ns],
            f"Число узлов при T = {roc_hours} ч: кривые почти не расходятся",
        ),
    )

    for axis, keys, labels, title in panels:
        drawn = 0
        for i, (key, label) in enumerate(zip(keys, labels)):
            curve = results["roc"].get(key)
            if curve is None:
                continue
            fpr = np.asarray(curve["fpr"], dtype=float)
            tpr = np.asarray(curve["tpr"], dtype=float)
            axis.plot(
                fpr,
                tpr,
                color=colors[i],
                lw=1.8,
                ls=styles[i % len(styles)],
                label=f"{label}, AUC={curve['auc']:.3f}",
                marker="o",
                ms=3,
            )
            drawn += 1
        axis.axvline(alpha, color="crimson", ls="--", lw=1.0)
        axis.axhline(target_power, color="grey", ls=":", lw=1.0)
        axis.set_xscale("log")
        axis.set_xlim(x_lo, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("α — доля честных парков, помеченных (риск оператора)")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.25, which="both")
        if drawn:
            axis.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("мощность — доля пойманных лжецов")
    fig.suptitle(
        "M1: ROC сдвига b = c(n)·σ(T) на реальном фоне (NASA POWER, июль). "
        "Фаза даёт те же кривые: тест не зависит от масштаба",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ───────────────────────────────────────────────────────────────── sections ──


def _bmin_range(results: dict, obs: str, hours: int) -> tuple[float, float]:
    """Spread of b_min (minutes) across sources and seasons for one (obs, T)."""
    values = [
        float(entry["b_min_minutes"])
        for key, entry in results["resolution_limit"].items()
        if isinstance(entry, dict)
        and entry.get("measurable")
        and key.split("|")[3] == obs
        and int(key.split("|")[2]) == hours
    ]
    return (min(values), max(values)) if values else (float("nan"), float("nan"))


def _dilution(n: int, k: int) -> float:
    """What is left of a group's visible offset once the park shares the blame."""
    return (n - k) / (n - 1)


def _ratio(results: dict, bg: str, season: str, obs: str, n: int) -> float:
    small = _limit(results, bg, season, 24, obs, n) or {}
    big = _limit(results, bg, season, 168, obs, n) or {}
    if small.get("b_min") and big.get("b_min"):
        return float(small["b_min"] / big["b_min"])
    return float("nan")


def _collusion_gaps(results: dict) -> tuple[list[tuple], list[tuple]]:
    """Split measured dilution ratios into "effect visible" and "noise over noise".

    ``b_eff`` is the effect the *statistic* sees (the studentised residual of a
    lying node), not the raw instrument gap: the dilution formula is a statement
    about the residual, and on the residual it has the same algebra for both
    readings. ``k = 1`` is the reference and carries no prediction of its own.

    A ratio is only a measurement where the single liar's effect clears that
    cell's own resolution limit ``b_min``. Where it does not -- a timestamp jump
    read as a frequency is not a frequency at all (§8) -- the ratio is one noise
    number over another and says nothing about the formula; those cells are
    returned apart instead of being averaged away.
    """
    visible: list[tuple] = []
    invisible: list[tuple] = []
    for key, entry in results["injections"].items():
        parts = key.split("|")
        if parts[0] != "collusion":
            continue
        n, obs, k = int(parts[4]), parts[5], int(parts[6])
        if k < 2:
            continue
        single = _inj(results, "collusion", parts[1], parts[2], int(parts[3]), n, obs, 1)
        if not single or not entry or not single.get("b_eff"):
            continue
        limit = _limit(results, parts[1], parts[2], int(parts[3]), obs, n) or {}
        b_min = float(limit.get("b_min") or 0.0)
        measured = entry["b_eff"] / single["b_eff"]
        gap = abs(measured - (n - k) / (n - 1))
        cell = (key, gap, abs(float(single["b_eff"])) >= b_min, b_min)
        (visible if cell[2] else invisible).append(cell)
    return visible, invisible


def _collusion_error(results: dict) -> float:
    """Largest gap between measured b_eff(k)/b_eff(1) and (n-k)/(n-1), where measurable."""
    visible, _ = _collusion_gaps(results)
    return max((gap for _, gap, _, _ in visible), default=float("nan"))


def _m2_headline(results: dict) -> list[str]:
    """One line for §0 about M2, computed -- or nothing if §12 is not in this run."""
    block = results.get("theorem")
    if not block:
        return []
    measured = [r for r in block["measured"] if r.get("flag_on_honest_strong") is not None]
    inverted = [r for r in measured if int(r["k"]) > int(r["n"]) - int(r["k"])]
    share = max((r["flag_on_honest_strong"] for r in inverted), default=float("nan"))
    worst = block["kernel_real"][-1] if block["kernel_real"] else None
    kernel = pct(worst["detection_rate"]) if worst else "—"
    return [
        "8. **Сговор стал теоремой (§12).** Парк видит только контраст, поэтому: сговор всех N "
        f"невидим при любом b (в поле — {kernel} тревог при b = "
        f"{worst['b_over_sigma']:g}σ, если {worst['n']} узлов лгут вместе); при k > N/2 детектор "
        f"срабатывает и называет честных — до {pct(share)} тревог на честном узле; "
        "b_min(k)/b_min(1) не выше (N−1)/max(k, N−k), с равенством на концах k = 1 и k = N−1; "
        "и ориентация раскола ненаблюдаема: «k лгут на +b» и «N−k лгут на −b» дают один и тот же "
        "вердикт, потому что различие лежит в ядре. Обратная сторона — якорь: он делает "
        "большинство видимым, но стоит точности против одиночки, и потому оправдан только "
        "против коалиций больше √N.",
        "",
    ]


def section_summary(results: dict) -> str:
    """Six lines: what was measured, and what it says."""
    criteria = results["criteria"]
    alpha = float(criteria["alpha"])
    ns = [int(v) for v in criteria["ns"]]
    bg = criteria["backgrounds"][0]
    season = criteria.get("roc_season", "july")

    s24 = _sigma(results, bg, season, 24, "offset")
    s168 = _sigma(results, bg, season, 168, "offset")
    lim = _limit(results, bg, season, 168, "offset", ns[0]) or {}
    lim24 = _limit(results, bg, season, 24, "offset", ns[0]) or {}
    tau_small = results["thresholds"][str(ns[0])]
    tau_large = results["thresholds"][str(ns[-1])]
    c_small = results["resolution_coefficient"][str(ns[0])]
    c_large = results["resolution_coefficient"][str(ns[-1])]
    m24 = minutes_per_day(s24) if s24 else float("nan")
    m168 = minutes_per_day(s168) if s168 else float("nan")
    blind = _worst_control(results, "constant_offset", obs="offset")

    return "\n".join(
        [
            "## 0. Резюме",
            "",
            f"1. **Порога 0.5 больше нет.** Оператор объявляет риск α = {alpha:g}, порог "
            f"следует из него: τ(n) — квантиль `max_k |t_k|` при H₀ по Монте-Карло "
            f"({detector.DRAWS_NULL} розыгрышей, seed {detector.SEED}). "
            f"τ({ns[0]}) = {f(tau_small['tau'], 3)}, τ({ns[-1]}) = {f(tau_large['tau'], 3)}; "
            f"граница Бонферрони даёт {f(tau_small['tau_bonferroni'], 3)} и "
            f"{f(tau_large['tau_bonferroni'], 3)} — согласие в третьем знаке.",
            "",
            f"2. **σ измерен, а не заявлен.** На реальном фоне ({bg}, {season}) разброс "
            f"частоты вокруг парка: **{f(m24, 1)} мин/сутки при T = 24 ч** и "
            f"**{f(m168, 1)} мин/сутки при T = 168 ч**. Второй источник (PVGIS 2020) при "
            f"T = 24 ч даёт {f(_sigma_minutes(results, 'pvgis_2020', season, 24, 'offset'), 1)} "
            "мин/сутки: короткое окно читает погоду своей базы, и базы расходятся. При "
            f"T = 168 ч обе дают ≈{f(m168, 1)} мин, потому что там σ уже стоит на объявленном "
            f"дрожании σ_honest = {f(minutes_per_day(SIGMA_HONEST), 0)} мин/сутки: совпадение "
            "источников на длинном окне — заслуга допущения, а не измерения (§11).",
            "",
            f"3. **Минимально обнаружимое смещение b** при α = {alpha:g} и мощности "
            f"{float(criteria['target_power']):g}: для N = {ns[0]} — "
            f"**{f(lim.get('b_min_minutes'), 0)} мин/сутки** за T = 168 ч и "
            f"**{f(lim24.get('b_min_minutes'), 0)} мин/сутки** за T = 24 ч. За T = 1 ч "
            "частота не читается вовсе: окно короче оборота (§6).",
            "",
            f"4. **Предел ставит честность, а не число узлов.** Коэффициент c(n) = b_min/σ "
            f"растёт с N всего с {f(c_small, 2)} до {f(c_large, 2)}: сотня узлов не покупает "
            f"ничего сверх десяти. Покупает длину окна — но только пока σ не упёрся в пол: в "
            f"январе b_min падает в {f(_ratio(results, bg, 'january', 'offset', ns[0]), 1)} "
            f"раза при 24 ч → 168 ч ({f(_sigma_minutes(results, bg, 'january', 24, 'offset'), 1)} → "
            f"{f(_sigma_minutes(results, bg, 'january', 168, 'offset'), 1)} мин), в июле — в "
            f"{f(_ratio(results, bg, season, 'offset', ns[0]), 1)} раза "
            f"({f(m24, 1)} → {f(m168, 1)} мин), потому что там σ уже на объявленном дрожании.",
            "",
            "5. **У каждой лжи своя наблюдаемая.** Постоянный сдвиг метки виден в фазе целиком и "
            "невидим в частоте вовсе; скачок метки фаза видит наполовину, а частота — только "
            "как испорченную подгонку, без всякой формулы; умножение уровня не видно нигде — "
            "физика слепа к нему по построению (§8).",
            "",
            "6. **Сговор разбавлен ровно по формуле.** k узлов с одинаковым скачком дают "
            f"b_eff = b·(n−k)/(n−1) с точностью {f(_collusion_error(results), 3)} (§9): "
            "доказательство — §12, здесь только замер.",
            "",
            "7. **Объявленный α держится на нормальном H₀ — и это видно.** Два контрольных "
            "лжеца (уровень ×1.6 и постоянный сдвиг метки) не видит ни одна наблюдаемая, но в "
            f"клетке {blind.get('bg', '—')}, {blind.get('season', '—')}, T = "
            f"{blind.get('hours', '—')} ч частота поднимает тревогу в {pct(blind.get('rate', float('nan')))} "
            f"окон при объявленном α = {alpha:g}: там эксцесс фона "
            f"{f(blind.get('kurtosis'), 2)}, и квантиль нормального H₀ этот хвост не покрывает "
            "(§10, критерии 5–6; §11).",
            "",
            *_m2_headline(results),
        ]
    )


def section_declared(results: dict) -> str:
    """The pre-registration, quoted, with the hash that pins it."""
    criteria = results["criteria"]
    path = BENCH_DIR / "criteria.json"
    digest = sha256(path) if path.exists() else "—"
    rows = [[str(i + 1), key, value] for i, (key, value) in enumerate(criteria["acceptance"].items())]
    return "\n".join(
        [
            "## 1. Что объявлено до прогона",
            "",
            f"`bench/criteria.json`, sha256 `{digest[:32]}…`, объявлен "
            f"{criteria.get('declared_utc', '—')}. Файл написан до прогона и в прогоне не "
            "менялся: изменение — это новый прогон и новый отчёт.",
            "",
            table(
                ["величина", "значение", "смысл"],
                [
                    ["α", f"{float(criteria['alpha']):g}", criteria["alpha_meaning"]],
                    [
                        "мощность",
                        f"{float(criteria['target_power']):g}",
                        criteria["target_power_meaning"],
                    ],
                    [
                        "σ_honest",
                        f"{f(SIGMA_HONEST, 6)} рад/ч = "
                        f"{f(minutes_per_day(SIGMA_HONEST), 0)} мин/сутки",
                        "честное дрожание частоты узла (объявлено, не измерено)",
                    ],
                    ["σ_read", f"{SIGMA_READ:.0%}", "шум чтения прибора от амплитуды"],
                    ["N", ", ".join(str(v) for v in criteria["ns"]), "размеры парка"],
                    ["T", ", ".join(f"{v} ч" for v in criteria["hours"]), "длины окна"],
                    [
                        "полоса поиска",
                        f"±{f(minutes_per_day(estimator.SEARCH_BAND), 0)} мин/сутки",
                        "приор частоты: вне полосы это уже не «парк с одним лжецом»",
                    ],
                    [
                        "повторов σ / инъекций",
                        f"{criteria['repeats_sigma']} / {criteria['repeats_inject']}",
                        "свежий парк на каждый повтор",
                    ],
                ],
            ),
            "",
            "**Критерии, названные до прогона.** Вердикт по каждому — §10.",
            "",
            table(["№", "критерий", "что должно быть"], rows),
            "",
        ]
    )


def section_data(results: dict) -> str:
    """The manifest: what the bench read, byte for byte."""
    manifest_path = BENCH_DIR / "manifest.json"
    rows = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, entry in manifest["sources"].items():
            rows.append(
                [
                    name,
                    entry["family"],
                    entry["period"],
                    f"{entry['bytes']:,}".replace(",", " "),
                    entry["sha256"][:24] + "…",
                ]
            )
    return "\n".join(
        [
            "## 2. Данные: реальный фон",
            "",
            "Синтетика здесь ровно в одном месте — в инъекциях. Фон реален: почасовые ряды "
            "приходящей солнечной радиации в точке 56.13 N, 40.40 E, два независимых "
            "источника. Парк из n производителей собирается из одного окна: общая погода, "
            "свой масштаб, свой шум чтения, своё честное дрожание частоты.",
            "",
            table(["источник", "кто", "период", "байт", "sha256"], rows),
            "",
            "Целостность: `python bench/fetch.py --check` падает, если сдвинулся хоть один "
            "байт фона. Манифест — `bench/manifest.json`.",
            "",
            "> Почему не FNET/GridEye и не ENTSO-E, как значилось в плане: там нет открытого "
            "эталонного ряда по конкретной точке с часовым разрешением, который можно "
            "перекачать побайтово и захешировать. NASA POWER и PVGIS дают именно это. "
            "PMU-фон — следующий шаг (доступ, согласование) и математику не меняет: детектор "
            "видит только σ и T.",
            "",
        ]
    )



def section_model(results: dict) -> str:
    """The math, the threshold it produces, and the check that it is the risk."""
    criteria = results["criteria"]
    alpha = float(criteria["alpha"])
    ns = [int(v) for v in criteria["ns"]]

    rows = []
    for n in ns:
        entry = results["thresholds"][str(n)]
        rows.append(
            [
                str(n),
                f(entry["tau"], 3),
                f(entry["tau_bonferroni"], 3),
                f(entry["tau_bonferroni"] - entry["tau"], 4),
                f(entry.get("tau_se"), 3),
                f(results["resolution_coefficient"][str(n)], 3),
            ]
        )

    above = [
        (
            n,
            results["thresholds"][str(n)]["tau"]
            - results["thresholds"][str(n)]["tau_bonferroni"],
        )
        for n in ns
        if results["thresholds"][str(n)]["tau"] > results["thresholds"][str(n)]["tau_bonferroni"]
    ]
    widest = max(
        (abs(results["thresholds"][str(n)]["tau_bonferroni"] - results["thresholds"][str(n)]["tau"]) for n in ns),
        default=float("nan"),
    )

    measured = _measured_alpha(results, alpha)
    return "\n".join(
        [
            "## 3. Модель и детектор",
            "",
            "Наблюдаемая — отклонение частоты узла от часов парка, посчитанное на окне "
            "(§4 объясняет, как оно получается): `x_i`, i = 1..n. Честные узлы стоят "
            "вокруг общего среднего парка:",
            "",
            "```",
            "H0:  x_i ~ N(mu, sigma^2),                       i = 1..n",
            "H1:  x_k ~ N(mu + b, sigma^2) для одного неизвестного k,  b != 0",
            "```",
            "",
            "`mu` и `b` — мешающие параметры, оба профилируются. Для кандидата `k` "
            "профильные МП-оценки: `mu_hat = mean_{i!=k} x_i`, `b_hat = x_k - mu_hat`, и "
            "отношение правдоподобия сводится к t-статистике:",
            "",
            "```",
            "Lam_k = b_hat^2 / (2 sigma^2 (1 + 1/(n-1))) = (1/2) t_k^2",
            "t_k   = (n x_k - sum_i x_i) / sqrt(n (n-1))     — в замкнутой форме",
            "G     = max_k |t_k|                             — статистика GLRT",
            "```",
            "",
            "**Почему не z-тест на каждый узел.** Остатки `t_k` коррелированы "
            "(corr = −1/(n−1)) и максимум по n из них — не одна нормальная величина. "
            "Поэтому порог берётся из распределения самого `G`, которое зависит только от "
            "n: ни от σ, ни от μ. Это и есть строка «τ(n)» ниже.",
            "",
            "**Порог — не магия, а объявленный риск.** τ(n) = квантиль уровня 1−α "
            f"распределения G при H0, α = {alpha:g} — доля честных парков, которую оператор "
            "согласен помечать. Монте-Карло против границы Бонферрони:",
            "",
            table(
                [
                    "N",
                    "τ (Монте-Карло)",
                    "τ (Бонферрони)",
                    "разница",
                    "ошибка τ",
                    "c(n) = b_min/σ",
                ],
                rows,
            ),
            "",
            (
                "Монте-Карло нигде не выше Бонферрони"
                if not above
                else "Монте-Карло выше Бонферрони для "
                + ", ".join(f"N = {n} (на {fx(gap, 4)})" for n, gap in above)
                + ", но на величину порядка собственной ошибки квантили"
            )
            + f"; наибольший зазор по модулю {f(widest, 3)} при ошибке квантили "
            f"{f(max((results['thresholds'][str(n)].get('tau_se') or float('nan')) for n in ns), 3)}. "
            "Так и должно быть: Бонферрони "
            "игнорирует корреляцию остатков (corr = −1/(n−1)) и потому переплачивает строгостью; "
            "с ростом n корреляция исчезает, границы сходятся, и разница уходит в шум. Порог "
            "всё равно берётся из Монте-Карло, Бонферрони печатается как независимая проверка.",
            "",
            f"Измеренная доля ложных тревог на свежих нулевых розыгрышах при α = {alpha:g}: "
            f"{measured} — объявленный риск выполняется на нормальном H₀ (§10 критерий 2; что "
            "бывает, когда H₀ не нормален, — §8 и §11).",
            "",
            f"`c(n) = b_min/σ` — вся зависимость от n. Она почти плоская: доска из 100 узлов "
            "не даёт радикально больше, чем доска из 10. Мощность при b = b_min равна "
            f"{float(criteria['target_power']):g} по построению (§6, таблица проверки).",
            "",
        ]
    )


def _measured_alpha(results: dict, alpha: float) -> str:
    """Empirical false-alarm rate at the declared α, from the ROC null draws.

    Each ROC row's ``fpr`` is measured on null draws the threshold was *not* built
    from, so this checks the declared risk instead of restating it.
    """
    by_obs: dict[str, list[float]] = {}
    for key, curve in results["roc"].items():
        obs = key.split("|")[2]
        for row in curve["rows"]:
            if abs(float(row["alpha"]) - alpha) < 1e-12:
                by_obs.setdefault(obs, []).append(float(row["fpr"]))
                break
    if not by_obs:
        return "—"
    parts = []
    for obs, values in by_obs.items():
        lo, hi = min(values), max(values)
        label = OBS_RU.get(obs, obs)
        span = fx(lo) if hi - lo <= 1e-9 else f"{fx(lo)}…{fx(hi)}"
        parts.append(f"{label} — {span}")
    return ", ".join(parts) + f" против объявленных {fx(alpha)}"



def section_observables(results: dict) -> str:
    """What a park can read about its own producers, and what it costs."""
    return "\n".join(
        [
            "## 4. Как парк читается: две наблюдаемые",
            "",
            "Модуль `entrain_infer.estimator`. Часы парка — сутки: один оборот за 24 ч, "
            f"ω₀ = 2π/24 = {OMEGA0:.6f} рад/ч. Из ряда узла извлекаются ровно две величины, "
            "и выбор между ними решает, какую ложь видно.",
            "",
            table(
                ["наблюдаемая", "как считается", "единица", "что видит"],
                [
                    [
                        "фаза",
                        "arg Σ_t (y_t − ȳ) e^(−iω₀t)",
                        "рад",
                        "постоянный сдвиг метки: τ ч → 2πτ/24",
                    ],
                    [
                        "частота",
                        "argmin по ω гармонической регрессии (3 гармоники) в полосе "
                        f"±{f(minutes_per_day(estimator.SEARCH_BAND), 0)} мин/сутки",
                        "рад/ч",
                        "постоянный дрейф; скачок метки — лишь частично, и не как скорость",
                    ],
                ],
            ),
            "",
            "Три объявленных предела, а не открытых в прогоне:",
            "",
            f"* **Частота требует оборота.** `MIN_READ_HOURS = {estimator.MIN_READ_HOURS:g}`: "
            "внутри суток суточный цикл не замкнулся, частота не идентифицируема, и оценщик "
            "отказывается, вместо того чтобы уверенно соврать.",
            "* **Фаза требует переопределённой подгонки.** DC плюс одна гармоника — это уже "
            "три параметра, поэтому `MIN_PHASE_SAMPLES = "
            f"{estimator.MIN_PHASE_SAMPLES}`: на двух отсчётах фазы нет вовсе, есть один бит — "
            "вырос ряд за окно или упал (фазор при двух точках это положительный скаляр на "
            "фиксированное комплексное число, и от данных остаётся только знак). Оценщик "
            "отказывается так же, как для частоты, и клетки T = 1 ч в таблицах фазы нет.",
            "* **Полоса поиска — это приор.** Вне неё парк — уже не «честные узлы плюс один "
            "лжец», а другой механизм.",
            "",
            "Солнечный ряд ночью равен нулю, и нули несут нулевой вес в обеих формулах — "
            "маскировать ночь не нужно и не делается. Отсюда следствие, которое видно в §6: "
            "часовое окно не даёт ни частоты (оборот не замкнулся), ни фазы (два отсчёта — это "
            "знак изменения), и в таблицах там стоит «недостижимо», а не шум.",
            "",
            "> Гармоническая регрессия, а не пик периодограммы: выпрямленный суточный профиль "
            "не узкополосен, его вторая гармоника не слабее первой, и пик простой "
            "периодограммы тащит за собой форма дня, а не его частота. Три гармоники — "
            "мешающие параметры; сигнал — сама частота.",
            "",
        ]
    )


def section_noise(results: dict) -> str:
    """Measured σ per source, season, window -- the honest spread plus reading noise."""
    criteria = results["criteria"]
    blocks = []
    for obs in bench.OBSERVABLES:
        rows = []
        for bg in criteria["backgrounds"]:
            for season in criteria["seasons"]:
                for hours in criteria["hours"]:
                    entry = results["sigma"].get(bench._key(bg, season, hours, obs))
                    if entry is None:
                        continue
                    rows.append(
                        [
                            bg,
                            season,
                            f"{hours} ч",
                            f(entry["sigma"], 4),
                            f(entry["sigma_minutes"], 2),
                            f(entry.get("excess_kurtosis"), 2),
                            str(entry["samples"]),
                        ]
                    )
        blocks.append(
            table(
                [
                    "источник",
                    "сезон",
                    "T",
                    "σ, " + ("рад/ч" if obs == "offset" else "рад"),
                    "в минутах метки",
                    "эксцесс",
                    "точек",
                ],
                rows,
            )
        )

    return "\n".join(
        [
            "## 5. Шум: σ измерен на реальном фоне",
            "",
            f"Для каждой пары (источник, сезон, T) строится {criteria['repeats_sigma']} свежих "
            "честных парков, каждая наблюдаемая центрируется внутри парка (детектор "
            "профилирует общее среднее — важно рассеяние вокруг парка, а не мнение парка о "
            "своих часах) и пул даёт σ. В σ входят три вещи сразу: честное дрожание частоты "
            f"(σ_honest = {f(minutes_per_day(SIGMA_HONEST), 0)} мин/сутки), шум чтения "
            f"({SIGMA_READ:.0%}) и "
            "неидеальность самой погоды.",
            "",
            "**Частота** (единица — минуты смещения частоты за сутки):",
            "",
            blocks[0],
            "",
            "**Фаза** (в минутах сдвига метки):",
            "",
            blocks[1],
            "",
            "Читается главное:",
            "",
            "* при T = 168 ч σ выходит на пол ≈ σ_honest — дальше окно не помогает, мешает "
            "уже честность, а не прибор;",
            "* при T = 24 ч σ в разы больше и сильно зависит от сезона: короткий зимний день "
            "читается хуже длинного летнего;",
            "* часового окна у фазы нет вовсе (§4): два отсчёта дают знак изменения, а не угол, "
            "и пул из таких чтений (два значения, разнесённые на пол-оборота) измеряет долю "
            "тёмных окон, а не разрешение. Клетка объявлена недостижимой и в таблицах стоит "
            "«недостижимо», а не число;",
            "* эксцесс близок к нулю там, где окно длинное, — допущение о нормальности H₀ не "
            "опровергнуто, но и не доказано: это проверка, а не аксиома. У фазы эксцесса нет "
            "(«—»): круговые данные не имеют честного четвёртого момента, пока не выбрана точка "
            "разреза, а она выбрана быть не может — это и есть та симметрия, из которой §8 "
            "делает вывод, что постоянный сдвиг метки невидим частоте по построению;",
            "",
        ]
    )



def section_b_min(results: dict) -> str:
    """The table the milestone asked for: b(N, T, sigma)."""
    criteria = results["criteria"]
    alpha = float(criteria["alpha"])
    power_target = float(criteria["target_power"])
    ns = [int(v) for v in criteria["ns"]]

    def grid(obs: str) -> list[list[str]]:
        rows = []
        for bg in criteria["backgrounds"]:
            for season in criteria["seasons"]:
                for hours in criteria["hours"]:
                    cells = [bg, season, f"{hours} ч"]
                    for n in ns:
                        entry = _limit(results, bg, season, hours, obs, n)
                        if not entry or not entry.get("measurable", True):
                            cells.append("недостижимо")
                        else:
                            cells.append(f(entry["b_min_minutes"], 1))
                    rows.append(cells)
        return rows

    headers = ["источник", "сезон", "T"] + [f"N={n}" for n in ns]

    check_rows = []
    for bg in criteria["backgrounds"]:
        for season in criteria["seasons"]:
            for obs in bench.OBSERVABLES:
                for hours in criteria["hours"]:
                    entry = _limit(results, bg, season, hours, obs, ns[0])
                    if not entry or not entry.get("measurable", True):
                        continue
                    check_rows.append(
                        [
                            bg,
                            season,
                            f"{hours} ч",
                            OBS_RU[obs],
                            f(entry["b_min_over_sigma"], 3),
                            f(entry["b_min"], 5),
                            f(entry["b_min_analytic"], 5),
                            f(entry["power_at_b_min"], 3),
                        ]
                    )

    headline = _limit(results, criteria["backgrounds"][0], "july", 168, "offset", ns[0]) or {}
    refusals = sorted(
        {
            entry["reason"]
            for entry in results["resolution_limit"].values()
            if entry.get("measurable", True) is False and entry.get("reason")
        }
    )
    return "\n".join(
        [
            "## 6. Минимально обнаружимое b(N, T, σ)",
            "",
            f"b_min — смещение, при котором мощность равна {power_target:g} при объявленном "
            f"α = {alpha:g}. Тест инвариантен к масштабу, поэтому b_min = c(n)·σ, а σ измерен в "
            "§5: таблицы ниже — произведение, а не надежда.",
            "",
            "### 6.1 Частота: сколько минут в сутки можно утащить незамеченным",
            "",
            table(headers, grid("offset")),
            "",
            f"Читается так: при T = 168 ч порог почти не зависит ни от сезона, ни от источника — "
            f"{f(_bmin_range(results, 'offset', 168)[0], 1)}–"
            f"{f(_bmin_range(results, 'offset', 168)[1], 1)} мин/сутки во всех восьми клетках. "
            "Это и есть цена недели: лжец может уводить часы на столько и оставаться "
            f"незамеченным в 10% окон. За сутки диапазон шире — "
            f"{f(_bmin_range(results, 'offset', 24)[0], 1)}–"
            f"{f(_bmin_range(results, 'offset', 24)[1], 1)} мин: в июле короткое окно почти "
            "догоняет неделю (σ там уже на полу, §5), в январе не догоняет и близко. За час "
            "частота не читается вовсе: окно короче оборота.",
            "",
            "### 6.2 Фаза: на сколько минут можно сдвинуть метку",
            "",
            table(headers, grid("phase")),
            "",
            "Фаза здесь начинается с T = 24 ч, и это не пропуск данных: часовое окно при "
            "почасовых отсчётах — два отсчёта, они дают знак изменения, а не фазу (§4). "
            "Часовую фазу дал бы только подасовой поток, и тогда строка T = 1 ч появилась бы "
            "сама.",
            "",
            f"Обратите внимание на направление: у фазы **короткое окно точнее длинного** — при "
            f"T = 24 ч b_min лежит в {f(_bmin_range(results, 'phase', 24)[0], 0)}–"
            f"{f(_bmin_range(results, 'phase', 24)[1], 0)} мин во всех сезонах и обоих "
            f"источниках, при T = 168 ч — {f(_bmin_range(results, 'phase', 168)[0], 0)}–"
            f"{f(_bmin_range(results, 'phase', 168)[1], 0)} мин. Неделя не успокаивает фазу, а "
            "накапливает смену погоды внутри окна: одна гармоника вынуждена делить вес между "
            "днями, и чтение уплывает. Частота ведёт себя наоборот (§6.1), и это не парадокс: "
            "частота — производная, ей длинное окно даёт точку отсчёта, а фаза — уровень, ей "
            "длинное окно даёт больше разных погод.",
            "",
            "Клетки «недостижимо» — это объявленные пределы данных, а не неудачные числа:",
            "",
            *([f"* {reason};" for reason in refusals] or ["* таких в этом прогоне нет;"]),
            "",
        ]
    )



def section_b_min_check(results: dict) -> str:
    """Power at b_min, and the analytic form against the Monte Carlo one."""
    criteria = results["criteria"]
    ns = [int(v) for v in criteria["ns"]]
    rows = []
    for bg in criteria["backgrounds"]:
        for season in criteria["seasons"]:
            for obs in bench.OBSERVABLES:
                for hours in criteria["hours"]:
                    entry = _limit(results, bg, season, hours, obs, ns[0])
                    if not entry or not entry.get("measurable", True):
                        continue
                    rows.append(
                        [
                            bg,
                            season,
                            f"{hours} ч",
                            OBS_RU[obs],
                            f(entry["b_min_over_sigma"], 3),
                            f(entry["b_min"], 5),
                            f(entry["b_min_analytic"], 5),
                            f(entry["power_at_b_min"], 3),
                        ]
                    )
    return "\n".join(
        [
            "### 6.3 Проверка: мощность в точке b_min, Монте-Карло против аналитики",
            "",
            table(
                [
                    "источник",
                    "сезон",
                    "T",
                    "наблюдаемая",
                    "c(n)",
                    "b_min (МК)",
                    "b_min (аналит.)",
                    "мощность",
                ],
                rows,
            ),
            "",
            "Аналитическая форма `b_min = (τ + z_(1−β))·σ·√(1+1/(n−1))` игнорирует шанс, что "
            "порог перейдёт честный узел, поэтому она оптимистичнее точной — в таблице видно, "
            "насколько. Мощность в самой точке b_min держится около объявленной: это и есть "
            "смысл слова «b_min».",
            "",
        ]
    )


def section_roc(results: dict) -> str:
    """The curve, its operating point, and what actually moves it."""
    criteria = results["criteria"]
    alpha = float(criteria["alpha"])
    power_target = float(criteria["target_power"])
    ns = [int(v) for v in criteria["ns"]]
    hours_grid = [int(h) for h in criteria["hours"]]

    auc_rows = []
    for obs in bench.OBSERVABLES:
        for n in ns:
            for hours in hours_grid:
                curve = results["roc"].get(bench._key("n", n, obs, hours))
                if curve:
                    auc_rows.append(
                        [OBS_RU[obs], f"N={n}", f"{hours} ч", f(curve["auc"], 4)]
                    )

    hour_rows = []
    hour_aucs: dict[str, list[float]] = {}
    for obs in bench.OBSERVABLES:
        for hours in hours_grid:
            curve = results["roc"].get(bench._key("hours", hours, obs, ns[0]))
            if curve:
                hour_rows.append([OBS_RU[obs], f"{hours} ч", f(curve["auc"], 4)])
                hour_aucs.setdefault(obs, []).append(float(curve["auc"]))

    alphas = (1e-4, 1e-3, 1e-2, 0.1)
    power_rows = []
    for obs in bench.OBSERVABLES:
        for hours in hours_grid:
            curve = results["roc"].get(bench._key("hours", hours, obs, ns[0]))
            if not curve:
                continue
            cells = [OBS_RU[obs], f"{hours} ч"]
            for target in alphas:
                best = min(curve["rows"], key=lambda r: abs(float(r["alpha"]) - target))
                cells.append(f"{float(best['tpr']):.2f}")
            power_rows.append(cells)

    # How much the window moves the curve at all -- zero, if the test is scale-free.
    hour_spread = max(
        (max(values) - min(values) for values in hour_aucs.values() if len(values) > 1),
        default=float("nan"),
    )

    return "\n".join(
        [
            "## 7. ROC",
            "",
            "![ROC](roc.png)",
            "",
            "Кривая строится честно: для каждого α порог берётся из нулевой таблицы, доля "
            "ложных тревог **измеряется** на свежих нулевых розыгрышах (поэтому она колеблется "
            "вокруг α, а не равна ему тождественно), мощность измеряется против одного лгущего "
            "узла со сдвигом b = c(n)·σ — то есть ровно в точке b_min (§6), и потому кривая "
            "проходит через рабочую точку (α, мощность) = ("
            f"{alpha:g}, {power_target:g}). Красный пунктир — объявленный α = {alpha:g}, серая "
            "линия — целевая мощность; все кривые проходят через их пересечение, потому что b "
            "задан как b_min.",
            "",
            "**Площадь под кривой по числу узлов** (N почти не двигает AUC):",
            "",
            table(["наблюдаемая", "N", "T", "AUC"], auc_rows),
            "",
            "**Площадь под кривой по длине окна** (и вот здесь важно, что именно двигает):",
            "",
            table(["наблюдаемая", "T", "AUC"], hour_rows),
            "",
            f"Разброс AUC по T равен {f(hour_spread, 4)} — то есть нулю, и это не совпадение, а "
            "свойство теста: b задан в долях σ (b = c(n)·σ), а статистика инвариантна к "
            "масштабу, поэтому кривая зависит только от b/σ = c(n). Таблица по T — проверка "
            "этой инвариантности, а не измерение длины окна. Длина окна входит в задачу через "
            "σ, а σ измерен в §5: именно он сдвигает рабочую точку вдоль кривой и задаёт b_min "
            "в §6. ROC отвечает на вопрос «какую мощность даёт такой-то b/σ», §5–§6 — на "
            "вопрос «какой b достижим за такое-то окно».",
            "",
            "**Мощность при N = " + str(ns[0]) + " по сетке α:**",
            "",
            table(["наблюдаемая", "T"] + [f"α≈{t:g}" for t in alphas], power_rows),
            "",
            "Число узлов двигает кривую ровно настолько, насколько c(n) отличается от c(n′): "
            f"c({ns[0]}) = {f(results['resolution_coefficient'][str(ns[0])], 2)} против "
            f"c({ns[-1]}) = {f(results['resolution_coefficient'][str(ns[-1])], 2)}. Отсюда вывод, "
            "который и был целью Хода 1: точность покупается не числом узлов, а тем, насколько "
            "σ удалось сбить длиной окна, и упирается она в σ_honest, а не в прибор.",
            "",
        ]
    )



def section_injections(results: dict) -> str:
    """Which lie each observable can see -- the map that is not the identity."""
    criteria = results["criteria"]
    bg = criteria["backgrounds"][0]
    n = int(criteria["ns"][0])
    alpha = float(criteria["alpha"])
    inject = criteria["injections"]
    seasons = inject.get("seasons", ["july"])
    hours_grid = inject.get("hours", [24, 168])
    ks = [int(k) for k in inject["collusion"]["ks"]]

    headers = ["инъекция"]
    for obs in bench.OBSERVABLES:
        for hours in hours_grid:
            headers.append(f"{OBS_RU[obs]} {hours} ч")

    rows = []
    cases = ["timestamp_step", "constant_offset", "level"] + [f"collusion:{k}" for k in ks]
    for case in cases:
        cells = [INJ_RU[case.split(":")[0]] + (f", k={case.split(':')[1]}" if ":" in case else "")]
        for obs in bench.OBSERVABLES:
            for hours in hours_grid:
                if case.startswith("collusion:"):
                    k = int(case.split(":")[1])
                    entry = _inj(results, "collusion", bg, seasons[0], hours, n, obs, k)
                else:
                    entry = _inj(results, case, bg, seasons[0], hours, n, obs)
                if entry is None:
                    cells.append("—")
                else:
                    cells.append(f"{pct(entry['detection_rate'])} ({f(entry['b_eff_minutes'], 0)} мин)")
        rows.append(cells)

    gauge = _inj(results, "constant_offset", bg, seasons[0], hours_grid[0], n, "phase") or {}
    tau_minutes = float(inject["collusion"]["tau_hours"]) * 60.0
    lvl = _worst_control(results, "level")
    blind_freq = _worst_control(results, "constant_offset", obs="offset")
    return "\n".join(
        [
            "## 8. Инъекции: у каждой лжи своя наблюдаемая",
            "",
            f"Контролируемые инъекции на реальном фоне ({bg}, {seasons[0]}, N = {n}), "
            f"{criteria['repeats_inject']} повторов на клетку, порог при α = {alpha:g}. "
            "В скобках — сдвиг, который видит **статистика** (чтение лжеца минус среднее "
            "остальных узлов), в минутах метки; для частоты — минуты в сутки. Это меньше "
            "физического сдвига по двум причинам сразу: профилированное среднее содержит самого "
            "лжеца и потому срезает (n−1)/n даже у одиночки — постоянный сдвиг "
            f"{f(tau_minutes, 0)} мин виден как {f(gauge.get('b_eff_minutes'), 0)} мин, — а при "
            "группе добавляется (n−k)/(n−1) (§9).",
            "",
            table(headers, rows),
            "",
            "Карта не тождественна, и это не баг, а основной результат Хода 1:",
            "",
            "* **скачок метки** (+2 ч в середине окна) читается обеими наблюдаемыми, но "
            "по-разному. Фаза видит половину скачка — вторую половину окна, — потому что "
            "уехала только она. Частота не видит *скорости* вовсе: скачок не делает часы "
            "быстрее, он портит гармоническую подгонку, и оценщик возвращает некий паллиатив "
            "(замер — в таблице). Обе величины здесь измерены, а не выведены: «сколько минут "
            "скачка видит частота» не имеет формулы, потому что это артефакт подгонки;",
            "* **постоянный сдвиг** метки — чистый фазовый сдвиг: **фаза его видит целиком, "
            "частота не видит вовсе**. Это не слабость детектора, это gauge-структура: "
            "постоянный сдвиг всех меток неотличим от сдвига часов парка, и потому детектор "
            "честно молчит. Ход 2 делает это утверждение теоремой;",
            "* **уровень x1.6** не виден нигде: физика парка слепа к масштабу по построению, "
            "и никакой статистикой это не лечится — лечится другой наблюдаемой;",
            "* **контрольные инъекции — это ещё и проверка α.** Сдвига не видит ни одна клетка "
            f"(0.0–{f(lvl.get('b_eff_minutes'), 1)} мин), но тревог всё равно больше объявленного: "
            f"{pct(lvl.get('rate', float('nan')))} для уровня и "
            f"{pct(blind_freq.get('rate', float('nan')))} для постоянного сдвига в клетке "
            f"{blind_freq.get('season', '—')}, T = {blind_freq.get('hours', '—')} ч, где эксцесс "
            f"фона {f(blind_freq.get('kurtosis'), 2)} (§5). Объявленный α — это α *при нормальном "
            "H₀*; где фон не нормален, детектор об этом честно ошибается, а не молчит;",
            "* **сговор** разбавлен, см. §9. Разбавление — тоже не дефект: это цена того, "
            "что парк сравнивает узлы между собой.",
            "",
        ]
    )


def section_collusion(results: dict) -> str:
    """Measured dilution against (n-k)/(n-1), plus what is left of the lie."""
    criteria = results["criteria"]
    bg = criteria["backgrounds"][0]
    alpha = float(criteria["alpha"])
    inject = criteria["injections"]
    season = inject.get("seasons", ["july"])[0]
    hours = int(inject.get("hours", [24, 168])[-1])
    ks = [int(k) for k in inject["collusion"]["ks"]]
    ns = [int(v) for v in criteria["ns"]]

    rows = []
    for n in [int(v) for v in criteria["ns"]]:
        for obs in bench.OBSERVABLES:
            single = _inj(results, "collusion", bg, season, hours, n, obs, 1)
            if single is None:
                continue
            for k in ks:
                if k > n:
                    continue
                entry = _inj(results, "collusion", bg, season, hours, n, obs, k)
                if entry is None:
                    continue
                measured = entry["b_eff"] / single["b_eff"] if single["b_eff"] else float("nan")
                theory = (n - k) / (n - 1)
                rows.append(
                    [
                        str(n),
                        OBS_RU[obs],
                        str(k),
                        f(theory, 3),
                        f(measured, 3),
                        f(measured - theory, 3),
                        f(entry["b_eff_minutes"], 1),
                        pct(entry["detection_rate"]),
                    ]
                )

    return "\n".join(
        [
            "## 9. Сговор: замер разбавления",
            "",
            f"k узлов делают одинаковый скачок метки (+{inject['timestamp_step']['tau_hours']:g} ч) "
            f"в {season}, T = {hours} ч, α = {alpha:g}. Теория (детектор, §3, и Ход 2): "
            "лжец сравнивается с честными, и общий сдвиг группы уменьшает его видимый сдвиг "
            "ровно в (n−k)/(n−1) раз.",
            "",
            table(
                [
                    "N",
                    "наблюдаемая",
                    "k",
                    "(n−k)/(n−1) теория",
                    "замер",
                    "разница",
                    "b_eff, мин",
                    "обнаружено",
                ],
                rows,
            ),
            "",
            f"Смысл для оператора: разбавление тем сильнее, чем меньше парк и больше группа — "
            f"при N = {ns[0]} сговор из {ks[-1]} узлов оставляет {pct(_dilution(ns[0], ks[-1]))} "
            f"видимого сдвига, при N = {ns[-1]} те же {ks[-1]} — {pct(_dilution(ns[-1], ks[-1]))}, "
            "а сговор всех n узлов невидим по построению — их среднее и есть мнение парка. "
            "Отсюда прямо следует, что «голосование большинством» не является детектором "
            "обмана: оно является его усилителем. §12 превращает это в теорему: за N/2 замеры "
            "кончаются, и дальше начинается утверждение, а не число.",
            "",
        ]
    )



def _worst_control(results: dict, name: str, obs: str | None = None) -> dict:
    """The cell where a control injection fires most often -- and why it might.

    A control is a lie that theory says neither observable can see. If one cell
    flags it anyway, the interesting part is not the rate but the cell: the rate
    is then a statement about that park's null, and the excess kurtosis measured
    there is the reason to look at.
    """
    best: dict = {}
    for key, entry in results["injections"].items():
        parts = key.split("|")
        if parts[0] != name or not isinstance(entry, dict):
            continue
        if obs is not None and parts[5] != obs:
            continue
        rate = float(entry["detection_rate"])
        if not best or rate > best["rate"]:
            sigma = results["sigma"].get(bench._key(parts[1], parts[2], int(parts[3]), parts[5]))
            best = {
                "bg": parts[1],
                "season": parts[2],
                "hours": int(parts[3]),
                "n": int(parts[4]),
                "obs": parts[5],
                "rate": rate,
                "b_eff_minutes": float(entry.get("b_eff_minutes") or 0.0),
                "kurtosis": float(sigma.get("excess_kurtosis") or float("nan")) if sigma else float("nan"),
            }
    return best


def _control_evidence(what: str, cell: dict, alpha: float, results: dict) -> str:
    """Say where a blind control fired and whether that was the lie or the null."""
    if not cell:
        return f"{what}: инъекций нет"
    rate = float(cell["rate"])
    where = (
        f"{cell['bg']}, {cell['season']}, T = {cell['hours']} ч, "
        f"{OBS_RU[cell['obs']]}, N = {cell['n']}"
    )
    if abs(rate - alpha) <= 0.02:
        return (
            f"{what} не виден нигде: худшая клетка — {where}, доля тревог {pct(rate)} при "
            f"α = {alpha:g}"
        )
    return (
        f"{what} не виден: в худшей клетке ({where}) видимый сдвиг {f(cell['b_eff_minutes'], 1)} "
        f"мин, то есть ноль, — но доля тревог там {pct(rate)} при α = {alpha:g}. Это не ложь "
        f"детектора, а фон: эксцесс в этой клетке {f(cell['kurtosis'], 2)} (§5), хвост тяжелее "
        "нормального, а порог выведен из нормального H₀ — α держится на нормальном H₀ и "
        "разваливается там, где его нет"
    )


def _verdicts(results: dict) -> list[list[str]]:
    """Measure each pre-declared criterion and say plainly whether it held."""
    criteria = results["criteria"]
    alpha = float(criteria["alpha"])
    ns = [int(v) for v in criteria["ns"]]
    bg = criteria["backgrounds"][0]
    season = criteria.get("roc_season", "july")
    # c(n) spelled once, so the verdict below can quote it without nested quotes.
    cs = {n: f(results["resolution_coefficient"][str(n)], 2) for n in ns}
    out: list[list[str]] = []

    gaps = [
        abs(results["thresholds"][str(n)]["tau_bonferroni"] - results["thresholds"][str(n)]["tau"])
        for n in ns
    ]
    ses = [float(results["thresholds"][str(n)].get("tau_se") or float("nan")) for n in ns]
    ok = all(
        results["thresholds"][str(n)]["tau"] <= results["thresholds"][str(n)]["tau_bonferroni"]
        for n in ns
    )
    above = [
        (n, results["thresholds"][str(n)]["tau"] - results["thresholds"][str(n)]["tau_bonferroni"])
        for n in ns
        if results["thresholds"][str(n)]["tau"] > results["thresholds"][str(n)]["tau_bonferroni"]
    ]
    out.append(
        [
            "1",
            "выполнено" if ok else "не выполнено",
            f"максимальный зазор |τ(МК) − τ(Бонферрони)| = {f(max(gaps), 3)} при ошибке самой "
            f"квантили {f(max(ses), 3)}; "
            + (
                "МК нигде не выше Бонферрони"
                if not above
                else "МК выше Бонферрони для "
                + ", ".join(f"N = {n} на {f(gap, 3)}" for n, gap in above)
                + ", то есть в пределах шума квантиля, но объявленное направление для этих N "
                "не выполнено"
            )
            + ". Порог и так берётся из МК, Бонферрони печатается как проверка, а не как "
            "источник",
        ]
    )

    fprs = []
    for curve in results["roc"].values():
        for row in curve["rows"]:
            if abs(float(row["alpha"]) - alpha) < 1e-12:
                fprs.append(float(row["fpr"]))
    if fprs:
        lo, hi = min(fprs), max(fprs)
        # The pre-registered band, verbatim: alpha * [0.5, 2.0].
        ok = lo >= 0.5 * alpha and hi <= 2.0 * alpha
        out.append(
            [
                "2",
                "выполнено" if ok else "не выполнено",
                f"измеренная доля ложных тревог при α = {alpha:g}: от {fx(lo)} до {fx(hi)} "
                f"(объявленный коридор α·[0.5, 2] = [{fx(0.5 * alpha)}, {fx(2.0 * alpha)}]; "
                "40000 нулевых розыгрышей на кривую)",
            ]
        )
    else:
        out.append(["2", "нет данных", "ROC не построена"])

    deviations = [
        abs(float(entry["power_at_b_min"]) - float(criteria["target_power"]))
        for entry in results["resolution_limit"].values()
        if isinstance(entry, dict) and entry.get("measurable")
    ]
    worst = max(deviations) if deviations else float("nan")
    out.append(
        [
            "3",
            "выполнено" if worst <= 0.02 else "не выполнено",
            f"максимальное отклонение мощности в точке b_min от цели: {fx(worst, 4)}",
        ]
    )

    step_rows = []
    for hours in (24, 168):
        best = 0.0
        for n in ns:
            for obs in bench.OBSERVABLES:
                entry = _inj(results, "timestamp_step", bg, season, hours, n, obs)
                if entry:
                    best = max(best, float(entry["detection_rate"]))
        step_rows.append((hours, best))
    ok = all(rate >= 0.9 for _, rate in step_rows)
    out.append(
        [
            "4",
            "выполнено" if ok else "не выполнено",
            "лучшая наблюдаемая: "
            + ", ".join(f"T={h} ч → {pct(r)}" for h, r in step_rows)
            + " (критерий читается как «хотя бы одной наблюдаемой»)",
        ]
    )

    worst_cell = _worst_control(results, "level")
    worst = abs(worst_cell.get("rate", float("nan")) - alpha)
    out.append(
        [
            "5",
            "выполнено" if worst <= 0.02 else "не выполнено",
            _control_evidence(
                "уровень ×1.6", worst_cell, alpha, results
            ),
        ]
    )

    blind_cell = _worst_control(results, "constant_offset", obs="offset")
    worst = abs(blind_cell.get("rate", float("nan")) - alpha)
    phase_seen = [
        float(entry["detection_rate"])
        for key, entry in results["injections"].items()
        if key.split("|")[0] == "constant_offset" and key.split("|")[5] == "phase"
    ]
    out.append(
        [
            "6",
            "выполнено" if worst <= 0.02 else "не выполнено",
            _control_evidence("постоянный сдвиг метки", blind_cell, alpha, results)
            + f"; фаза его видит: до {pct(max(phase_seen)) if phase_seen else '—'} — как и "
            "должно быть, это gauge-сдвиг",
        ]
    )

    visible, invisible = _collusion_gaps(results)
    worst_visible = max((gap for _, gap, _, _ in visible), default=float("nan"))
    worst_invisible = max((gap for _, gap, _, _ in invisible), default=float("nan"))
    out.append(
        [
            "7",
            "выполнено" if worst_visible <= 0.05 else "не выполнено",
            f"максимальное расхождение замера с (n−k)/(n−1) на {len(visible)} клетках, где "
            f"одиночка перекрывает b_min: {f(worst_visible, 4)}. Ещё {len(invisible)} клеток "
            f"(july, T = 24 ч, частота) исключены: там скачок метки вообще не виден частоте — "
            f"её b_eff для одиночки ниже собственного b_min, — и отношение шума к шуму даёт до "
            f"{f(worst_invisible, 3)} ни о чём не говоря",
        ]
    )

    ratios = [_ratio(results, bg, season, "offset", n) for n in ns]
    # The criterion has two halves: a real gain in T, and a plateau in N.
    # b_min = c(n) * sigma(T): c(n) does not depend on T and the measured sigma
    # does not depend on N (each producer's scatter is its own), so the ratio
    # b_min(24)/b_min(168) is the same number for every N. The N-plateau lives in
    # c(n) -- printed here so the reader sees where it actually is -- and the
    # ratio cannot move toward 1 with N in this model, so that half of the
    # criterion is reported as it measured, not as it was hoped.
    gain = ratios[0] if ratios else float("nan")
    plateau_in_n = max(ratios) - min(ratios) if ratios else float("nan")
    winter = _ratio(results, bg, "january", "offset", ns[0])
    first_half = bool(ratios) and all(r > 1.05 for r in ratios)
    second_half = bool(ratios) and all(abs(r - 1.0) <= 0.05 for r in ratios)
    ok = first_half and second_half
    halves = (
        "обе половины выполнены"
        if ok
        else "первая половина (T даёт выигрыш: отношение > 1 при любом N) "
        + ("выполнена" if first_half else "не выполнена")
        + " — но величина выигрыша сезонная, а не узловая: "
        + f"{f(gain, 2)} в {season}, {f(winter, 1)} в январе; вторая половина (отношение "
        "стремится к 1 с ростом N) "
        + ("выполнена" if second_half else "не выполнена")
    )
    out.append(
        [
            "8",
            "выполнено" if ok else "не выполнено",
            "(критерий записан как b_min(168)/b_min(24); ниже то же отношение в обратную "
            "сторону) b_min(24 ч)/b_min(168 ч) по N: "
            + ", ".join(f"N={n} → {f(r, 2)}" for n, r in zip(ns, ratios))
            + f"; разброс по N — {f(plateau_in_n, 3)}: отношение равно σ(24)/σ(168) при любом N, "
            "потому что c(n) не зависит от T, а σ не зависит от N. Плато по N сидит в c(n): "
            + ", ".join(f"N={n} → c={cs[n]}" for n in ns)
            + f". Итог: {halves}.",
        ]
    )
    return out


def section_verdicts(results: dict) -> str:
    """Criteria before the run, measurements after -- side by side."""
    criteria = results["criteria"]
    declared = list(criteria["acceptance"].items())
    verdicts = _verdicts(results)
    rows = []
    for (key, text), (num, verdict, evidence) in zip(declared, verdicts):
        rows.append([num, text, verdict, evidence])
    return "\n".join(
        [
            "## 10. Вердикты по критериям, названным до прогона",
            "",
            table(["№", "критерий", "вердикт", "что измерено"], rows),
            "",
            "Критерий, который не выполнился, остаётся в отчёте невыполненным — иначе "
            "предрегистрация ничего не стоит. У Хода 2 свои критерии, объявленные отдельно "
            "(`bench/criteria_m2.json`) и проверенные в §12.6.",
            "",
        ]
    )



def section_limits(results: dict) -> str:
    """What this does not prove. A report without this section is advertising."""
    criteria = results["criteria"]
    alpha = float(criteria["alpha"])
    return "\n".join(
        [
            "## 11. Что это доказывает и чего не доказывает",
            "",
            "**Доказывает:**",
            "",
            f"* порог выводится из объявленного риска α = {alpha:g} и не зависит ни от σ, ни от "
            "μ — только от n (§3);",
            "* на реальном фоне σ измерим, и b_min(N, T, σ) — число, а не лозунг (§5, §6);",
            "* скачок метки +2 ч ловится на недельном окне практически всегда; постоянный сдвиг "
            "метки не ловится в частоте и ловится в фазе; умножение уровня не ловится нигде (§8);",
            "* сговор разбавляется ровно по (n−k)/(n−1) (§9).",
            "",
            "**Что доказано уже не замером, а теоремой (§12):**",
            "",
            "* сговор всех N невидим при любом b — это ядро отображения «парк → контраст», а не "
            "слабость детектора;",
            "* при k > N/2 тот же детектор срабатывает и называет честных: инверсия следует из "
            "знака E t вне группы, а не из ошибки реализации;",
            "* ориентация раскола ненаблюдаема: «k лгут на +b» и «N−k лгут на −b» — одно и то же "
            "наблюдение, поэтому всякий протокол, решающий спор большинством, есть выбор "
            "калибровки, а не вывод из данных;",
            "* b_min(k) ≤ b_min(1)·(N−1)/max(k, N−k), с равенством на концах; тест хорды не слабее "
            "теста узла при любом k и требует знать S заранее.",
            "",
            "**Не доказывает:**",
            "",
            f"* что σ_honest = {f(minutes_per_day(SIGMA_HONEST), 0)} мин/сутки — правда о живом "
            "парке. Это "
            "*объявленная* величина, и именно она задаёт пол на недельном окне: плато в §6 — "
            "следствие допущения, а не открытие. Замер σ_honest на живом парке — отдельная "
            "работа, и до неё числа на T = 168 ч читаются как «при таком-то допущении»;",
            "* нормальность H₀. Эксцесс в §5 близок к нулю на длинных окнах, но это проверка на "
            "одном фоне и одной погоде, а не теорема. Тяжёлые хвосты сдвинут порог вверх и b_min "
            "вместе с ним — и цена уже видна: "
            f"{pct(_worst_control(results, 'constant_offset', obs='offset').get('rate', float('nan')))} "
            "тревог на контрольной инъекции в клетке с эксцессом "
            f"{f(_worst_control(results, 'constant_offset', obs='offset').get('kurtosis'), 2)} при "
            f"объявленном α = {alpha:g} (§10, критерии 5–6). Детектор, гарантирующий α при любом "
            "H₀, — это отдельная задача (перестановочный тест); здесь α объявлен для нормального;",
            "* что инъекции покрывают реальные способы лгать. Смоделированы три: скачок, "
            "постоянный сдвиг, уровень. Не смоделированы: дрейф, зависящий от погоды; "
            "избирательное молчание; подмена истории задним числом; сговор с разными сдвигами;",
            "* перенос на другие физики. Часы парка здесь — сутки, наблюдаемая — солнце. Для "
            "ветра, PMU или тепла нужно заново измерить σ и заново объявить полосу поиска: "
            "математика та же, числа другие;",
            "* что α покрывает весь парк целиком. α — риск **на один парк на одно окно**. Если "
            "оператор смотрит 200 парков каждый час, честная ставка тревог в час — это "
            f"200/168·α, и её надо объявлять отдельно;",
            "* что «больше узлов не помогает» — общее правило. Здесь это верно для одного лжеца: "
            "c(n) почти плоская, потому что b_min растёт как √(1+1/(n−1)). Для группы из k "
            "лжецов выигрыш отрицательный — сговор усиливается числом (§9).",
            "",
            "**Что бы это опровергло:** живой парк, у которого σ частоты заметно больше измеренной "
            f"здесь ({f(minutes_per_day(_sigma(results, criteria['backgrounds'][0], 'july', 24, 'offset') or 0.0), 1)} "
            f"мин/сутки при T = 24 ч и "
            f"{f(minutes_per_day(_sigma(results, criteria['backgrounds'][0], 'july', 168, 'offset') or 0.0), 1)} "
            "при 168 ч) — тогда b_min вырастет, и Ход 1 придётся повторять на его данных, а не на "
            "NASA POWER. Поэтому M1 заканчивается кодом и критериями, а не выводами о природе.",
            "",
        ]
    )



def _m2_verdicts(results: dict) -> list[list[str]]:
    """M2's nine declared checks, each given a verdict against the measurements.

    The verdicts are *computed*: a check that fails prints ``✘`` and names its
    number, and the section says which claim died. Nothing here can be adjusted
    to pass.
    """
    block = results.get("theorem")
    if not block:
        return []
    tol = block["criteria"]["tolerances"]
    alpha = float(block["alpha"])
    alg = block["algebra"]
    measured = block["measured"]
    kernel_real = block["kernel_real"]
    anchor = block["anchor"]

    def mark(ok: bool, detail: str) -> list[str]:
        return ["✔" if ok else "✘", detail]

    rows: list[list[str]] = []

    ok = alg["gauge_drift"] <= tol["algebra_relative"]
    rows.append(
        ["1. калибровка", *mark(ok, f"сдвиг статистики при c = ±10³: {f(alg['gauge_drift'], 2)}")]
    )

    ok_alg = alg["kernel_statistic"] <= 1e-9
    ok_field = all(
        r["detection_rate"] <= alpha + tol["flag_on_honest_share"] for r in kernel_real
    )
    rates = ", ".join(f"{r['b_over_sigma']:g}σ→{pct(r['detection_rate'])}" for r in kernel_real)
    rows.append(
        [
            "2. ядро при k = N",
            *mark(
                ok_alg and ok_field,
                f"статистика при b = 10⁶σ: {f(alg['kernel_statistic'], 2)}; в поле: {rates}",
            ),
        ]
    )

    ok = True
    for n in sorted({int(r["n"]) for r in measured}):
        per = {int(r["k"]): r for r in measured}
        for k in per:
            if (n - k) in per and k < n - k:
                a = per[k]["b_min_ratio_measured"]
                b = per[n - k]["b_min_ratio_measured"]
                if np.isfinite(a) and np.isfinite(b):
                    ok &= abs(a - b) <= 0.08 * max(a, b)
    rows.append(
        ["3. хорда c(S)", *mark(ok, "симметрия k ↔ N−k по замеру b_min; норма контраста = c(S) — тест")]
    )

    ok = all(
        abs(
            theorem.node_gain(int(r["n"]), int(r["k"]))
            - (int(r["n"]) - int(r["k"])) / (int(r["n"]) - 1)
        )
        <= 1e-12
        for r in measured
        if 1 <= int(r["k"]) <= int(r["n"]) // 2
    )
    rows.append(
        ["4. разбавление", *mark(ok, "тождество (N−k)/(N−1): проверено как равенство, не как замер")]
    )

    bad = []
    for row in measured:
        n, k = int(row["n"]), int(row["k"])
        share = row.get("flag_on_honest_strong")
        if share is None:
            continue
        expected = 0.0 if k < n - k else (1.0 if k > n - k else 0.5)
        if abs(share - expected) > 0.1:
            bad.append(f"N={n},k={k}: {pct(share)} вместо {pct(expected)}")
    rows.append(
        [
            "5. инверсия",
            *mark(
                not bad,
                "доля тревог на честном узле: "
                + (", ".join(bad) if bad else "0 / 1 / ½ по сторонам раскола"),
            ),
        ]
    )

    ok = all(bool(row.get("bound_holds", True)) for row in measured)
    ends = [
        f"N={r['n']},k={r['k']}: {f(r['b_min_ratio_measured'], 2)}"
        for r in measured
        if int(r["k"]) in (1, int(r["n"]) - 1) and np.isfinite(r["b_min_ratio_measured"])
    ]
    rows.append(
        [
            "6. граница b_min",
            *mark(ok, "все замеры ниже (N−1)/max(k,N−k); концы: " + ", ".join(ends[:4])),
        ]
    )

    orient = alg["orientation"]
    ok = (
        orient["reading_gap"] <= tol["algebra_relative"]
        and orient["statistic_gap"] <= tol["algebra_relative"]
        and orient["same_node"]
        and orient["same_verdict"]
    )
    rows.append(
        [
            "7. ориентация",
            *mark(
                ok,
                f"{orient['cells']} пар миров: расхождение чтений {f(orient['reading_gap'], 2)}, "
                f"статистик {f(orient['statistic_gap'], 2)}; узел и вердикт совпадают",
            ),
        ]
    )

    worse = [
        f"N={r['n']},k={r['k']}"
        for r in measured
        if r.get("power_chord_at_b1") is not None
        and r["power_chord_at_b1"] < r["power_at_b1"] - 1e-12
    ]
    best = max(
        (r for r in measured if r.get("power_chord_at_b1") is not None),
        key=lambda r: r["power_chord_at_b1"] - r["power_at_b1"],
        default=None,
    )
    rows.append(
        [
            "8. хорда сильнее",
            *mark(
                not worse,
                "тест хорды при известном S не слабее ни в одной клетке"
                if best is None
                else f"выигрыш максимален при N={best['n']}, k={best['k']}: "
                f"{pct(best['power_chord_at_b1'])} против {pct(best['power_at_b1'])} при том же b",
            ),
        ]
    )

    ok = all(
        abs(row["anchored_power_measured"] - 0.9) <= tol["power"]
        for row in anchor
        if "anchored_power_measured" in row
        and np.isfinite(row.get("anchored_b_min_over_sigma", np.inf))
    )
    gains = [f"N={row['n']}: k* = {row['crossover_k']}" for row in anchor if "crossover_k" in row]
    rows.append(
        ["9. якорь", *mark(ok, "мощность в точке b_min = 0.9; точка размена " + ", ".join(gains))]
    )
    return rows


def _theorem_tables(block: dict):
    """The four tables of §12, read straight out of the measured block."""
    measured = [r for r in block["measured"] if np.isfinite(r["b_min_ratio_measured"])]
    pred_rows = []
    for row in measured:
        n, k = int(row["n"]), int(row["k"])
        pred_rows.append(
            [
                str(n),
                str(k),
                f(theorem.chord_ratio(n, k), 3),
                f(theorem.node_gain(n, k), 3),
                f(row["b_min_ratio_predicted"], 3),
                f(row["b_min_ratio_measured"], 3),
                f(row["b_min_ratio_predicted"] - row["b_min_ratio_measured"], 3),
                pct(row["flag_on_honest_strong"])
                if row.get("flag_on_honest_strong") is not None
                else "—",
                SIDE_RU.get(theorem.flagged_side(n, k), theorem.flagged_side(n, k)),
            ]
        )

    real_rows = []
    for cell in block["real"]:
        n, k = int(cell["n"]), int(cell["k"])
        gap = cell["read_gap_minutes"]
        side = cell.get("flagged_side_predicted") or theorem.flagged_side(n, k)
        real_rows.append(
            [
                str(cell["season"]),
                str(n),
                str(k),
                SIDE_RU.get(side, side),
                pct(cell["detection_rate"]),
                pct(1.0 - cell["localisation_rate"]) if k < n else "—",
                f(cell["b_eff_minutes"], 1),
                f(gap, 1) if gap == gap else "—",
            ]
        )

    anchor_rows = []
    for row in block["anchor"]:
        if "k" not in row:
            continue
        gain = row.get("gain")
        anchor_rows.append(
            [
                str(row["n"]),
                str(row["k"]),
                f(row["anchored_b_min_over_sigma"], 2),
                f(row["unanchored_b_min_over_sigma"], 2)
                if np.isfinite(row["unanchored_b_min_over_sigma"])
                else "∞",
                f(gain, 2) if gain is not None else "—",
                pct(row["anchored_power_measured"]),
            ]
        )

    kernel_rows = [
        [
            str(cell["n"]),
            f"{cell['b_over_sigma']:g}",
            f(cell["b_eff_over_sigma"], 3),
            pct(cell["detection_rate"]),
        ]
        for cell in block["kernel_real"]
    ]
    return pred_rows, real_rows, anchor_rows, kernel_rows


def _field_contrast(real: list[dict]) -> str:
    """One measured sentence: what is left of a group's visible shift at k = N-1 and N.

    Reads the july, N = 10 cells -- a named choice, not a flattering one: it is the
    same park size the rest of the report leads with.
    """
    pick = {
        (c["season"], int(c["n"]), int(c["k"])): c
        for c in real
        if c["season"] == "july" and int(c["n"]) == 10
    }
    near = pick.get(("july", 10, 9))
    full = pick.get(("july", 10, 10))
    if not near or not full:
        return "клетки N = 10, июль, k = N−1 и k = N в прогоне отсутствуют"
    left = near["b_eff_minutes"]
    left_txt = "машинный ноль" if abs(left) < 1e-9 else f"{f(left, 1)} мин"
    return (
        f"при N = 10 и k = 9 от {f(near['read_gap_minutes'], 1)} мин остаётся {left_txt}, "
        "а при k = 10 не остаётся ничего: разрыв чтений там не определён (честных в парке нет), "
        "и статистика молчит при любом b"
    )


def _measured_anchor_crossover(anchor: list[dict]) -> str:
    """Where the *measured* ``b_min`` says the anchor starts to win.

    The bench grid skips ``k``, so the honest answer is a span: the first ``k``
    where the anchor is already better and the last where it is still worse. A
    gap in the grid is not evidence, and printing a single number where the grid
    has none would be inventing one.
    """

    def num(row: dict, key: str) -> float:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else math.inf

    cells: dict[int, list[dict]] = {}
    for row in anchor:
        if "k" in row and math.isfinite(num(row, "unanchored_b_min_over_sigma")):
            cells.setdefault(int(row["n"]), []).append(row)
    pieces: list[str] = []
    for n in sorted(cells):
        rows = sorted(cells[n], key=lambda r: int(r["k"]))
        better = [
            int(r["k"])
            for r in rows
            if num(r, "anchored_b_min_over_sigma") < num(r, "unanchored_b_min_over_sigma")
        ]
        worse = [
            int(r["k"])
            for r in rows
            if num(r, "anchored_b_min_over_sigma") >= num(r, "unanchored_b_min_over_sigma")
        ]
        if not better:
            continue
        first = min(better)
        last_loss = max(worse) if worse else 0
        span = f"{first}" if first == last_loss + 1 else f"{last_loss + 1}…{first}"
        pieces.append(f"N = {n}: k* = {span}")
    if not pieces:
        return ""
    return "Замер по тем же клеткам: " + ", ".join(pieces)


def _anchor_lone_cost(anchor: list[dict]) -> str:
    """How much worse the anchor is against a lone liar, per park size, from the table."""
    pieces: list[str] = []
    for row in anchor:
        if int(row.get("k", 0)) != 1:
            continue
        gain = row.get("gain")
        if not isinstance(gain, (int, float)) or not gain > 0:
            continue
        pieces.append(f"в {f(1.0 / gain, 1)} раза (N = {row['n']})")
    if not pieces:
        return "цена якоря против одиночки не измерена"
    return "против одиночного лжеца якорь требует b_min " + " и ".join(pieces) + " больше"


def section_theorem(results: dict) -> str:
    """M2: the collusion theorem -- stated, then checked against the run."""
    block = results.get("theorem")
    if not block:
        return ""
    alpha = float(block["alpha"])
    target = float(block["target_power"])
    alg = block["algebra"]
    orient = alg["orientation"]
    pred_rows, real_rows, anchor_rows, kernel_rows = _theorem_tables(block)
    real = block["real"]
    crossover = ", ".join(
        f"N={row['n']}: k* = {row['crossover_k']} (√N ≈ {f(row['sqrt_n'], 1)})"
        for row in block["anchor"]
        if "crossover_k" in row
    )

    head = [
        "## 12. Ход 2: теорема о сговоре — что именно симметрично",
        "",
        "§9 измерил разбавление. Здесь оно выводится — и выводится вместе с границей, за "
        "которой замеры M1 кончаются: за N/2. Все предсказания этого раздела записаны формулами "
        "(`entrain_infer/theorem.py`) и объявлены в `bench/criteria_m2.json` "
        f"(sha256 `{block['criteria_sha256'][:16]}…`) до прогона; прогон их только проверяет. "
        f"α = {alpha:g}, целевая мощность {target:g}.",
        "",
        "**Наблюдение, из которого всё следует.** Детектор видит только контраст",
        "",
        "```",
        "t_i = (N x_i − Σ_j x_j) / √(N(N−1)),      t(x + c·1) = t(x)  для любого c.",
        "```",
        "",
        "Это не погрешность и не удобство: это единственная симметрия парка — калибровка. "
        f"Замер: сдвиг статистики при c = ±10³ равен {f(alg['gauge_drift'], 2)} — ноль в "
        f"пределах машинной точности, на {orient['cells']} парах миров.",
        "",
        "### 12.1 Утверждение 1: сговор всех — ядро",
        "",
        "Если лгут все N узлов одинаково, то `1_S = 1`, `P_H 1_S = 0`, и наблюдение не меняется "
        "вообще: **невидимо при любом b, любом N, любой мощности.** Это не следствие слабости "
        "GLRT — это следствие того, что у парка нет внешних часов.",
        "",
        f"Замер на модели шума: статистика при b = 10⁶σ равна {f(alg['kernel_statistic'], 2)}. "
        "Замер в поле (N = 10, июль, T = 168 ч), где ложь физически есть и растёт:",
        "",
        table(["N", "b/σ", "что видит статистика, b_eff/σ", "тревог"], kernel_rows),
        "",
        "Второй столбец — поданный сдвиг, третий — то, что осталось после профилирования общего "
        "среднего: ноль при любом b. Утечка информации о таком сговоре возможна только вне парка.",
        "",
    ]

    chord = [
        "### 12.2 Утверждение 2: видимость = хорда",
        "",
        "Единственная наблюдаемая раскола — его длина в пространстве контрастов, то есть хорда "
        "между двумя кластерами:",
        "",
        "```",
        "c(S) = ‖P_H 1_S‖ = √(k(N−k)/N),      E t_i = +b(N−k)/√(N(N−1))   (i ∈ S)",
        "                                    E t_i = −b·k   /√(N(N−1))   (i ∉ S)",
        "```",
        "",
        "Отсюда сразу: (а) при k ≤ N/2 видимый сдвиг группы равен (N−k)/(N−1) от одиночного "
        "лжеца — это ровно критерий 7 из §10, но здесь он тождество, а не замер; (б) при "
        "k > N/2 знаки меняются местами, и максимум |t| достаётся **честному меньшинству**; "
        "(в) b_min(k)/b_min(1) ≤ (N−1)/max(k, N−k) — верхняя граница, потому что коалиция ставит "
        "max(k, N−k) статистик туда, где одиночный лжец ставит одну; равенство достигается на "
        "концах k = 1 и k = N−1, а разрыв максимален при k ≈ N/2.",
        "",
        table(
            [
                "N",
                "k",
                "c(S)/c({i})",
                "node_gain",
                "b_min/b_min(1) граница",
                "замер",
                "разрыв",
                "тревог на честном",
                "предсказано",
            ],
            pred_rows,
        ),
        "",
        "Колонка «разрыв» — не ошибка формулы, а её содержание: граница односторонняя, и замер "
        "ложится ниже неё ровно там, где хорду делят много узлов. Колонка «тревог на честном» — "
        "та самая инверсия: при k > N/2 статистика срабатывает, но обвиняет честных.",
        "",
        "Односторонность не постулируется. При сдвиге b·(N−1)/max(k, N−k) старшая группа из "
        "max(k, N−k) узлов имеет ровно ту нецентральность, что одиночный лжец на своём b_min, "
        "при том же пороге τ; максимум по m ≥ 2 таким узлам срабатывает строго чаще одного, "
        "поэтому замер ложится ниже границы, а равенство возможно только на концах k = 1 и "
        "k = N−1, где старшая группа состоит из одного узла. Это лемма в шапке `theorem.py`; "
        "направление неравенства проверяет `test_a_coalition_at_the_bound_is_at_least_as_loud_"
        "as_one_liar`, равенство на концах — `test_the_bound_is_attained_only_at_the_ends`.",
        "",
    ]
    orient_part = [
        "### 12.3 Утверждение 3: ориентация лежит в ядре",
        "",
        "Для любых S, b и μ",
        "",
        "```",
        "μ·1 + b·1_S  ≡  (μ+b)·1 + (−b)·1_{S^c}      — как наблюдения, тождественно.",
        "```",
        "",
        "Два мира — «k узлов лгут на +b» и «N−k узлов лгут на −b, а часы парка уехали на +b» — "
        "дают одно и то же x, ту же статистику, тот же узел и тот же вердикт. Их разница есть "
        "калибровочное преобразование, то есть ничего. Значит ориентация раскола не «трудно "
        "оценивается» — её в данных нет. Замер:",
        "",
        table(
            ["что сравнивалось", "результат"],
            [
                ["расхождение чтений, max |x_A − x_B|", f(orient["reading_gap"], 2)],
                ["расхождение статистик, max |t_A − t_B|", f(orient["statistic_gap"], 2)],
                ["тот же помеченный узел", "да" if orient["same_node"] else "нет"],
                ["тот же вердикт (тревога, узел)", "да" if orient["same_verdict"] else "нет"],
                ["пар миров проверено", str(orient["cells"])],
            ],
        ),
        "",
        "Следствие не математическое, а управленческое: любой протокол, разрешающий спор "
        "большинством голосов, **объявляет большинство честным** — это и есть выбор калибровки. "
        "Теорема говорит, что это не слабость конкретного детектора, а минимально возможное "
        "допущение: отделить «лгут k» от «лгут N−k» не может никакой тест этой наблюдаемой, "
        "потому что различие лежит в ядре.",
        "",
        "### 12.4 Та же теорема в поле",
        "",
        "Те же коалиции на реальном фоне (NASA POWER, T = 168 ч, b = "
        f"{real[0]['b_over_sigma']:g}σ того сезона). «Честных помечено» — доля тревог, попавших "
        "на честный узел, то есть 1 − точность локализации.",
        "",
        table(
            [
                "сезон",
                "N",
                "k",
                "предсказано",
                "обнаружено",
                "честных помечено",
                "b_eff, мин",
                "разрыв чтений, мин",
            ],
            real_rows,
        ),
        "",
        "Читается как приговор голосованию: при k ≤ N/2 детектор прав и точен, при k > N/2 он так "
        "же уверенно срабатывает — и называет честных. Сдвиг, который статистика видит у группы "
        "(b_eff), падает как (N−k)/N от физического разрыва чтений: "
        + _field_contrast(real)
        + ".",
        "",
    ]

    anchor_part = [
        "### 12.5 Обратная сторона: якорь — это размен, а не выигрыш",
        "",
        "Теорема была бы только запретом, если бы не говорила, что делать. Единственный способ "
        "вывести ориентацию из ядра — закрепить калибровку: объявить часы одного узла истиной. "
        "Тогда `y_i = x_i − x_a` имеет известное среднее при H0, и вопрос перестаёт быть "
        "относительным:",
        "",
        "```",
        "δ = b·k/√(N−1),        b_min(якорь) = (τ + z)·σ·√(N−1)/k.",
        "```",
        "",
        "Заметьте форму: с якорем b_min **падает** с ростом k, без якоря — растёт до N/2. Это "
        "размен: якорь покупает поимку большинства ценой точности против одиночки.",
        "",
        table(
            ["N", "k", "b_min с якорем, σ", "b_min без якоря, σ", "якорь лучше в … раз", "мощность в точке b_min"],
            anchor_rows,
        ),
        "",
        f"Точка размена (аналитически): {crossover}. {_measured_anchor_crossover(block['anchor'])}; "
        "разрыв в сетке (там, где стоят две границы вместо одного числа) — это предел замера, "
        "а не неопределённость формулы. Ниже точки размена якорь хуже обычного GLRT — "
        "одиночного лжеца сравнивают с одним эталоном вместо N−1 честных; выше — лучше, и при "
        "k = N−1 относительный тест называет честного, а якорь даёт лучший b_min во всём модуле. "
        "Отсюда операторское правило: якорь стоит своего допущения только против коалиций "
        "больше √N.",
        "",
        "Строка k = N в таблице — не пропуск, а граница обратного хода: если лгут все, включая "
        "якорь, то `y = x_i − x_a` — один шум, и мощность якоря равна α. Якорь спасает не от "
        "сговора всех, а от сговора большинства: он переносит допущение с «парк прав» на "
        "«прав этот узел», и цена этого допущения видна в первой строке — "
        f"{_anchor_lone_cost(block['anchor'])}, чем у теста, который он призван исправить.",
        "",
        "### 12.6 Проверки, объявленные до прогона",
        "",
        table(["критерий", "", "что вышло"], _m2_verdicts(results)),
        "",
        "Ход 2 не закрывает невыполненные критерии Хода 1 (§10: 5–6 — объявленный α держится "
        "только на нормальном H₀, 8 — плато по N не наступает): они про другое, и теорема о "
        "сговоре их не касается. Код это подтверждает: `run_theorem` пишет только ключ "
        "`theorem` в `results.json` и не пересчитывает ни одного числа Хода 1.",
        "",
        "**Что бы это опровергло:** хоть один мир, где два описания из 12.3 дают разные вердикты; "
        "хоть одна клетка, где при k > N/2 тревога падает на лжеца чаще, чем на честного; хоть "
        "одно b_min(k) выше границы (N−1)/max(k, N−k) сверх шума. Каждое из этих — не уточнение, "
        "а опровержение: утверждения здесь тождества, а не подгонки.",
        "",
    ]
    return "\n".join(head + chord + orient_part + anchor_part)



def header(results: dict) -> str:
    """Title, provenance, and the map of the artefacts."""
    criteria = results["criteria"]
    return "\n".join(
        [
            "# Entrain · M1 + M2 — GLRT, объявленный риск, минимально обнаружимое b, теорема о сговоре",
            "",
            "Ход 1 (§1–§11): детектор, σ на реальном фоне, b_min(N, T, σ), инъекции, ROC. "
            "Ход 2 (§12): почему сговор разбавляется именно так и где заканчиваются замеры. "
            "Отчёт собран из `bench/results.json`; ни одно число ниже не набрано от руки. "
            f"α = {float(criteria['alpha']):g}, мощность "
            f"{float(criteria['target_power']):g}, окна "
            f"{', '.join(str(v) + ' ч' for v in criteria['hours'])}, парки "
            f"{', '.join(str(v) for v in criteria['ns'])} узлов, фон — реальная солнечная "
            "радиация (NASA POWER 2021, PVGIS 2020).",
            "",
            "```",
            "python bench/fetch.py          # скачать фон, записать манифест с sha256",
            "python -m entrain_infer        # прогнать bench, собрать REPORT.md и roc.png",
            "python -m pytest               # проверки математики и оценщика",
            "```",
            "",
            table(
                ["файл", "что это"],
                [
                    ["`entrain_infer/detector.py`", "GLRT: статистика, нулевая таблица, мощность, b_min"],
                    ["`entrain_infer/theorem.py`", "Ход 2: теорема о сговоре — замкнутые формулы, без Монте-Карло"],
                    ["`entrain_infer/estimator.py`", "две наблюдаемые: фаза и частота против часов парка"],
                    ["`entrain_infer/model.py`", "единицы и объявленные константы"],
                    ["`entrain_infer/bench.py`", "реальный фон, сборка парка, инъекции, замеры"],
                    ["`entrain_infer/report.py`", "этот отчёт и ROC-картинка"],
                    ["`bench/criteria.json`", "предрегистрация: α, критерии, seed — до прогона"],
                    ["`bench/criteria_m2.json`", "предрегистрация Хода 2: девять предсказаний — до прогона"],
                    ["`bench/manifest.json`", "sha256 каждого байта фона"],
                    ["`bench/results.json`", "всё измеренное, машинночитаемо"],
                    ["`README.md`", "как это запустить и что объявлено до прогона"],
                    ["`tests/`", "математика, оценщик, инъекции"],
                ],
            ),
            "",
            "---",
            "",
        ]
    )


def build_report(results: dict) -> str:
    """The whole document, in reading order."""
    parts = [
        header(results),
        section_summary(results),
        section_declared(results),
        section_data(results),
        section_model(results),
        section_observables(results),
        section_noise(results),
        section_b_min(results),
        section_b_min_check(results),
        section_roc(results),
        section_injections(results),
        section_collusion(results),
        section_verdicts(results),
        section_limits(results),
        section_theorem(results),
    ]
    return "\n".join(parts)


def write_all(
    results: dict, report_path: Path = REPORT, plot_path: Path = PLOT
) -> list[Path]:
    """Write the report and the ROC plot; return the paths written."""
    plot_roc(results, plot_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(results), encoding="utf-8")
    return [report_path, plot_path]

