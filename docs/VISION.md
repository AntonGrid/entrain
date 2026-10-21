# Axis VISION — Proof of Reality

**The physical layer cryptography cannot provide.**
One page. Every number below names the command that reproduces it.

---

## English (one-pager)

### The problem: signatures prove authorship, not reality

DePIN's weakest link is not the key — it is the sensor. Ed25519 inside an NXP SE050
proves **who** sent a reading and that its bytes were not altered: `device_id`,
`nonce`, `ts`, `energy_wh`. It says nothing about whether the reading is physically
real. A compromised or simulated device signs a perfect forgery — valid key, valid
nonce, correct daily total, wrong hour.

Measured example (declared park, 8 producers, 168 h, 3 % noise): a **+2 h timestamp
lie**. The signature is valid, the daily energy total is unchanged, and the level
rules cannot see it. The phase reading sees it: **score 0.634** at threshold 0.5.
Cryptography answers *who*; the physics answers *when*.

Sensor spoofing is not a bug in a signature scheme. It is a category the scheme
does not cover.

### The solution: a physical layer (L0.5)

Entrain describes a medium of coupled oscillators (Kuramoto). Each producer is an
oscillator whose state is a **frequency**; what two producers have in common is
written into the **link** between them. Agreement is measured, not asserted:

```
R = |mean e^{iθ}| ∈ [0, 1]
```

`R` is not a model opinion. It is a number with a criterion declared **before** the
run, a reference implementation in Julia, and a Python port that matches it
**bit for bit** (divergence 0.0 on all 11 functions and all 3 law steps).

L0.5 is the *role*, not a new stack layer: in the ecosystem map the code lives in
**L3 (`ENRG-AI/agent/physics/`)**, but it answers a question that sits between L0
("who signed") and L1 ("does the proof pass the rules") — *is this physically
consistent with the park?*

### How it runs in Axis

| Step | What happens | Evidence |
|---|---|---|
| 1. Read | the park's own oracle proofs; besides energy, the **phase and frequency** of the daily cycle | same proofs the oracle already collects — no new sensor required |
| 2. Compute | `agent/physics/`: 8 modules, 1,804 lines; four readings per producer — `silence`, `resonance`, `phase`, `memory` → `score ∈ [0,1]` | 11 functions ported from `entrain/julia/kuramoto/basic.jl`, divergence **0.0** |
| 3. Gate | Policy Engine (ADR-0003), the single admissibility point: with **operator opt-in** `enforce_ai_anomaly` and `score ≥ 0.6` → `allowed=False`, `reason="ai_anomaly_flagged"` — the proof is not admitted, no mint follows | the gate already existed; **0 new policy flags, 0 protocol changes** |
| 4. Scale | pair coupling `link/N` (full graph), `link/z` (city); memory `(link + N₀·cos)/N` and `(link + N₀·cos)/z` | `link_c = 0.1915`, independent of N; total pull `link + N₀·⟨cos⟩` is O(1): **7.54** full graph vs **7.52** city (z = 4) at T = 1200 s |

The flag is **off by default**, and the gate is off-chain — deliberately not
mirrored on-chain. Physics advises; the operator or the DAO decides. Nothing in the
protocol changed: same envelope, same signature, same reason code.

### Scale: the law of a million

- Pair strength falls as `1/N`; total pull on one oscillator does not:
  `link + N₀·⟨cos⟩` is O(1). The pair dissolves, the sum stands.
- Threshold `link_c = 2γσ√(2π)/π = 0.1915` (γ = 0.8, σ = 0.15) is a property of the
  **pull per oscillator**, not of N. At `link = 0.6` we are **3.13× above it**;
  measured R = **0.978** at N = 100 (300 s run).
- City (z = 4): coupling `link/z` keeps the pull constant at O(1) cost per device —
  N·z links. Measured R = 0.641 (N = 100) and 0.400 (N = 200) at `link = 0.60`:
  a locally coherent city need not share one global rhythm. That is locality, not
  memory failure.
- **86/86 assertions** pass in three configurations: N = 10 (full graph), N = 30
  (full graph), N = 100 with z = 4 (city). The city is 22.6 min, N = 30 is 31.8 min
  on one core — a nightly check, not a CI step.

Extrapolation to 10⁶ devices is a **prediction of the formula**, not a run. We say
so in the book, in the code, and here.

### Our honesty

We do not sell "absolute protection". We sell a falsifiable claim and a map of its
limits.

| Claim | Number | How to check |
|---|---|---|
| Description is self-consistent | **86/86** assertions × 3 configurations | `julia julia/kuramoto/consistency.jl`, `city.jl`, `beyond.jl` |
| Julia reference = Python port | divergence **0.0** | `ENRG-AI/tests/test_physics_oscillator.py` against `entrain_reference.json` |
| Catches phase attacks, no false alarms | precision **1.00**, **0** false alarms, 3 of 4 declared changes | `python -m agent.physics.benchmark` |
| Honest about what it misses | level ×1.6 is invisible **by construction** (score 0.179) | same command; the rule detector needs a larger step |
| Test suites | **510 green** (227 Python L3 + 158 Python L1 + 125 Rust L2), 2 intentionally skipped | `pytest`, `cargo test` |

**Boundaries we publish ourselves:** collusion is invisible (if everyone lies
consistently, R is high — that is the method, not an oversight); there is no live
data yet (all numbers are from a declared model, seeds 7/11/23/42); the reading is
daily (a step in `t0` is half-read at `t0 + 12 h`); this is a **reading of reports**,
not a physical measurement — becoming a measurement requires a sensor such as
irradiance; and the metric does not affect minting, settlement or ERS by default.

---

## Русская версия

### Проблема: подпись доказывает авторство, а не реальность

Слабое место DePIN — не ключ, а датчик. Ed25519 внутри NXP SE050 доказывает, **кто**
прислал отсчёт и что его байты не подменены: `device_id`, `nonce`, `ts`,
`energy_wh`. О том, реальны ли эти числа физически, подпись не говорит ничего.
Скомпрометированное или смоделированное устройство подписывает безупречную
подделку: ключ верный, nonce верный, сумма за сутки верная, час — нет.

Измеренный пример (объявленный парк, 8 производителей, 168 ч, 3 % шума): **подмена
метки времени на +2 ч**. Подпись валидна, сумма энергии за сутки не изменилась,
правила уровня её не видят. Фазовое чтение видит: **score 0.634** при пороге 0.5.
Криптография отвечает «кто», физика отвечает «когда».

Sensor spoofing — не дефект схемы подписи. Это категория, которую схема не
покрывает.

### Решение: физический слой (L0.5)

Entrain описывает среду связанных осцилляторов (Курамото). Каждый производитель —
осциллятор, состояние которого есть **частота**; то, что у двух производителей
общего, записывается в **связь** между ними. Согласованность измеряется, а не
объявляется:

```
R = |среднее e^{iθ}| ∈ [0, 1]
```

`R` — не мнение модели. Это число с критерием, объявленным **до** прогона, с
эталоном на Julia и портом на Python, который совпадает с эталоном **до последнего
бита** (расхождение 0.0 на всех 11 функциях и всех 3 шагах закона).

L0.5 — имя **роли**, а не нового слоя стека: в карте экосистемы код живёт в
**L3 (`ENRG-AI/agent/physics/`)**, но отвечает на вопрос, стоящий между L0 («кто
подписал») и L1 («проходит ли пруф правила»): *согласовано ли это физически с
парком?*

### Как это работает в Axis

| Шаг | Что происходит | Чем подтверждено |
|---|---|---|
| 1. Чтение | пруфы оракула, которые уже есть; помимо энергии — **фаза и частота** суточного оборота | новый датчик не нужен |
| 2. Счёт | `agent/physics/`: 8 модулей, 1 804 строки; четыре чтения на производителя — `silence`, `resonance`, `phase`, `memory` → `score ∈ [0,1]` | 11 функций из `entrain/julia/kuramoto/basic.jl`, расхождение **0.0** |
| 3. Гейт | Policy Engine (ADR-0003), единственная точка допуска: при **включённом оператором** флаге `enforce_ai_anomaly` и `score ≥ 0.6` → `allowed=False`, `reason="ai_anomaly_flagged"` — пруф не допущен, минта по нему нет | гейт уже существовал: **0 новых флагов, 0 изменений протокола** |
| 4. Масштаб | связь на пару `link/N` (полный граф), `link/z` (город); память `(link + N₀·cos)/N` и `(link + N₀·cos)/z` | `link_c = 0.1915`, не зависит от N; полная тяга `link + N₀·⟨cos⟩` есть O(1): **7.54** полный граф против **7.52** город (z = 4) при T = 1200 с |

Флаг **по умолчанию выключен**, а гейт — off-chain: на цепи он сознательно не
зеркалится. Физика предлагает, оператор или DAO решает. В протоколе не изменилось
ничего: тот же конверт, та же подпись, тот же код причины.

### Масштаб: закон миллиона

- Сила пары падает как `1/N`, полная тяга на одну вибрацию — нет:
  `link + N₀·⟨cos⟩` есть O(1). Пара растворяется, сумма стоит.
- Порог `link_c = 2γσ√(2π)/π = 0.1915` (γ = 0.8, σ = 0.15) — свойство **тяги на
  вибрацию**, а не N. При `link = 0.6` мы **в 3.13 раза выше** него; измерено
  R = **0.978** при N = 100 (прогон 300 с).
- Город (z = 4): связь `link/z` держит тягу постоянной при цене O(1) на устройство —
  N·z связей. Измерено R = 0.641 (N = 100) и 0.400 (N = 200) при `link = 0.60`:
  локально согласованный город не обязан иметь один общий ритм. Это локальность,
  а не потеря памяти.
- **86/86 утверждений** сходятся в трёх конфигурациях: N = 10 (полный граф), N = 30
  (полный граф), N = 100 при z = 4 (город). Город — 22.6 мин, N = 30 — 31.8 мин на
  одном ядре: это ночная сверка, а не шаг CI.

Экстраполяция на 10⁶ устройств — **предсказание формулы**, а не прогон. Так
сказано и в книге, и в коде, и здесь.

### Наша честность

Мы не продаём «абсолютную защиту». Мы продаём опровержимое утверждение и карту его
границ.

| Утверждение | Число | Как проверить |
|---|---|---|
| Описание согласно с собой | **86/86** утверждений × 3 конфигурации | `julia julia/kuramoto/consistency.jl`, `city.jl`, `beyond.jl` |
| Эталон Julia = порт Python | расхождение **0.0** | `ENRG-AI/tests/test_physics_oscillator.py` против `entrain_reference.json` |
| Ловит фазовые атаки без ложных | точность **1.00**, **0** ложных, 3 из 4 объявленных изменений | `python -m agent.physics.benchmark` |
| Честно о том, что не ловит | уровень ×1.6 невидим **по построению** (score 0.179) | та же команда; правилу нужен больший шаг |
| Наборы тестов | **510 зелёных** (227 Python L3 + 158 Python L1 + 125 Rust L2), 2 пропущено осознанно | `pytest`, `cargo test` |

**Границы, которые мы называем сами:** сговор невидим (если все врут согласованно,
R высок — это свойство метода, а не недосмотр); живых данных пока нет (все числа —
на объявленной модели, сиды 7/11/23/42); чтение суточное (шаг в `t0` читается
наполовину через 12 ч); это **чтение отчётов**, а не физическое измерение — чтобы
стало измерением, нужен датчик (например, освещённости); и по умолчанию метрика не
влияет ни на минт, ни на расчёт, ни на ERS.

### Что дальше

1. **Пилот на живом парке** — ≥ 7 дней, ≥ 2 устройств; пересчитать пороги
   (0.5 и четыре шкалы) на живых данных и сказать об этом.
2. **Датчик освещённости** — чтобы чтение отчётов стало измерением среды.
3. **Причинный фильтр** — чтобы отвечать в онлайне, а не по последним 24 часам.
4. **Прогон города при N > 200** — чтобы экстраполяция на 10⁶ опиралась на прогон,
   а не только на формулу.

### Ссылки

| Что | Где |
|---|---|
| граница «физика ≠ протокол», полный протокол замера | `docs/ENTRAIN-BOUNDARY.md` |
| книга (онтология, главы 22 и 24) | `book/ENTRAIN.md` |
| математика: 86 утверждений, город, граница N | `julia/kuramoto/consistency.jl`, `city.jl`, `beyond.jl` |
| интеграция в L3 | `ENRG-AI/agent/physics/README.md`, `ENRG-AI/docs/PHYSICS_INTEGRATION_SUCCESS.md` |
| гейт политики | `Axis-core/docs/policy-engine.md`, `axis_core/policy/engine.py` |

Лицензия: Apache-2.0.
