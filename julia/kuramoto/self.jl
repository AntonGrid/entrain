# Entrain. Курамото: λ — не производная, а корень.
#
# В главе 22 и в `tension.jl` стоит признанное место без числа. Формула нужной
# жёсткости
#
#     λ* = F/требование − γ,     F = |D_p + γ·r₀_p − p·D₁|/(p·ω₁)
#
# читается как производная: тяга делится на требование. Но тяга F снята с того
# состояния, которое само живёт при λ — а λ как раз и ищется. Значит справа стоит
# функция от λ, и написанное равенство читается неверно: верное чтение — не
# «λ равна F/треб − γ», а
#
#     λ  ≥  g(λ),   g(λ) = (наибольшая F при жёсткости λ)/(требование) − γ,
#
# то есть множество держащих жёсткостей есть {λ : λ ≥ g(λ)}, а нужная жёсткость —
# **наименьшее** такое λ. Это и есть перевод в символы: λ — не производная, а
# корень неравенства; у формулы, читаемой как равенство, есть наклон, и всё
# зависит от него: если g растёт медленнее λ — порог есть и он один; если быстрее
# — держать нечем ни при какой жёсткости.
#
# Снятое при одном λ предсказание (так оно снято в `tension.jl`: при 16 и при 48)
# — это одно значение g, а не λ*. Отсюда и «предсказание идёт впереди измерения»:
# оно взято не в своей точке. Здесь это и проверяется: g читается при своём λ, и
# итерация λ ← g(λ) доводится до предела.
#
# Пороги объявлены до прогона, без порогов после.
#
#   1. Существование. Итерация от λ₀ = 16 даёт |g(λ*) − λ*| < 1e-6 не позже 40
#      шагов, и λ* ниже потолка шага (λ+γ)·dt = 2, то есть 199.2 при шаге 0.01,
#      у всех шести зёрен и трёх требований (10 %, 2 %, 1 %). Не так — у
#      самосогласования нет решения, и «нужная жёсткость» не число.
#   2. Единственность. Итерации от λ₀ = 1 и λ₀ = 128 сходятся к одному числу
#      (расхождение < 1 %). К разным — чисел больше одного, и текст «нужная
#      жёсткость» надо будет называть иначе.
#   3. Совпадение с держанием. |λ* − λ*(лестница)| ≤ 25 % у пяти зёрен из шести.
#      Больше — гипотеза мертва: самосогласование ведёт не туда, где аккорд
#      держится, и причина у расхождения не в чужом λ.
#   4. Чужое λ. Предсказание g(16) против λ*: если λ* не ближе к измеренному, чем
#      g(16), то «впереди измерения» объясняется не чужой точкой снятия, и
#      причину надо назвать числом, а не оговоркой.
#   5. Куда врёт формула. Для ступеней, где аккорд держится на деле (промах не
#      хуже требования), знак λ − g(λ); для ступеней, где не держится, — тот же
#      знак. Это отделяет «формула пессимистична» от «формула оптимистична» и
#      называет виновную часть.
#
# Первое чтение вскрыло у самой меры слабость: в этой среде аккорд написан с
# промахом 1 %, и требование «промах не хуже 1 %» выполняется уже тогда, когда
# тяга мала и обертон не сдвинулся, — держание там неотличимо от «не тронули».
# Поэтому второе чтение идёт на невырожденной мере: аккорд сбит на 20 %
# (`detune = 0.2`, как в `tension.jl`), требование 1 % строго ниже написанного, и
# держаться значит сжимать. Порог второго чтения — шестой:
#
#   6. Невырожденная мера. У пяти зёрен из шести итерация от 1 и от 16 сходится
#      (|Δλ| < 1e-6 не позже 60 шагов) к одному числу (расхождение < 1 %), и это
#      число укладывается в 25 % от измеренного лестницей. Не так — причина не в
#      вырожденной мере, а в самом равновесии: λ не корень и там.
#
# Уравнения те же, что в `basic.jl`. Ничего не подгоняется: тяга, промах и
# написанное снимаются с состояния, жёсткость задаётся и ищется; шаг — тот же
# `DT`. Числа печатаются по шести зёрнам размахом: где свойство, а где бросок
# генератора, видно по размаху.
#
# Запуск:
#
#     julia julia/kuramoto/self.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6
const NEEDS = (0.10, 0.02, 0.01)      # требования: промах не хуже, в долях
const GAMMA = 0.8                     # затухание: то же, что в `tension.jl`, `written.jl`
const BAND = 0.1                      # полоса: та же среда
const LINK = 0.6
const LADDER = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0,
                40.0, 48.0, 64.0, 96.0, 128.0)
const START = 16.0                    # откуда идёт итерация и откуда снято чужое λ
const FAR = START * 3                 # второе чужое λ: 48, как в `tension.jl`
const TOL = 1e-6
const STEPS = 40
const CEIL = 2 / DT - GAMMA           # потолок шага: (λ + γ)·dt < 2
const DETUNE = 0.2                    # сбитый аккорд: второе чтение, невырожденная мера
const FINE = (4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0, 56.0,
              64.0, 80.0, 96.0)       # лестница для сбитого аккорда: как `FINE20`


# --- что читается ---------------------------------------------------------

"""
    tensioned(; seed, tuning)

Среда того же вида, что в `tension.jl`: три моды, связь 0.6, полоса 0.1, память
1e-3, затухание 0.8. Написанное — то, чем пишет `build`: обертоны стоят с
промахом 1 %, и он у каждой среды свой.
"""
function tensioned(; seed::Int = 1, tuning::Float64 = 0.0, detune::Float64 = 0.0)
    m = build(modes = 3, band = BAND, link = LINK, tuning = tuning,
              damping = GAMMA, memory = 1e-3, seed = seed)
    if detune != 0.0
        rng = MersenneTwister(seed + 1000)
        for i in 1:size(m.omega0, 1), p in 2:size(m.omega0, 2)
            m.omega0[i, p] *= 1 + detune * randn(rng)
        end
        m.omega .= m.omega0
    end
    return m
end

"Промах обертонов: наибольшее |ω_p − p·ω₁|/(p·ω₁) по всем вибрациям и обертонам."
function x_over(m::Medium)
    worst = 0.0
    for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        worst = max(worst, abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base))
    end
    return worst
end

"Тяга связи на моду (i, p) — тот же член, что складывается в `step!`."
function drag(m::Medium, i::Int, p::Int)
    d = 0.0
    for j in 1:size(m.omega, 1)
        j == i && continue
        for n in 1:size(m.omega, 2)
            d += m.coupling[i, j] * overlap(m.omega[j, n] - m.omega[i, p], m.band) *
                 sin(m.theta[j, n] - m.theta[i, p])
        end
    end
    return d
end

"То, против чего держит строй: |D_p + γ·r₀_p − p·D₁| — числитель равновесия."
function held_force(m::Medium, i::Int, p::Int)
    r0 = m.omega0[i, p] - p * m.omega0[i, 1]
    return abs(drag(m, i, p) + m.damping * r0 - p * drag(m, i, 1))
end

"""
    probe(need, seed; lam)

Одно чтение при заданной жёсткости: среда проживает минуту, за которую снимается
**наибольшая** сила по строке (сила гуляет — фазы идут), и по ней считается
предсказание

    pred = F/(требование · p·ω₁) − γ

по тем строкам, где оно наибольшее. Это и есть g(λ) — правая часть равенства при
своём λ. Ровно так сила снимается и в `tension.jl` (`peak_pred_seed`), только там
λ чужое. Тут же промах через минуту: он говорит, держится ли аккорд на деле.
Сорвавшееся состояние — не «держится» и не «гаснет», а отметка: читать нечего.
"""
function probe(need::Float64, s::Int; lam::Float64, detune::Float64 = 0.0)
    m = tensioned(seed = s, tuning = lam, detune = detune)
    vibs, modes = size(m.omega)
    peaks = zeros(vibs, modes)
    broke = false
    for n in 1:round(Int, 60.0 / DT)
        step!(m, DT)
        if !(all(isfinite, m.omega) && all(isfinite, m.theta))
            broke = true
            break
        end
        n % round(Int, 0.5 / DT) == 0 || continue
        for i in 1:vibs, p in 2:modes
            peaks[i, p] = max(peaks[i, p], held_force(m, i, p))
        end
    end
    miss = broke ? Inf : x_over(m)
    pred, at = -Inf, (0, 0)
    for i in 1:vibs, p in 2:modes
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        v = peaks[i, p] / (need * p * base) - m.damping
        v > pred && ((pred, at) = (v, (i, p)))
    end
    return (pred = pred, miss = miss, broke = broke, at = at)
end

"Cached readings: (требование, зерно, λ, сбитость) → probe; один прогон на место."
const PROBES = Dict{Tuple{Float64,Int,Float64,Float64}, NamedTuple}()
function at(need::Float64, s::Int, lam::Float64; detune::Float64 = 0.0)
    key = (need, s, lam, detune)
    haskey(PROBES, key) && return PROBES[key]
    return PROBES[key] = probe(need, s; lam = lam, detune = detune)
end

"""
    iterate_need(need, s; start)

Итерация λ ← g(λ) от заданного начала: путь и предел. Три исхода названы
заранее: сошлось (|Δ| < TOL), «без строя» (предел не выше нуля — тогда
требование выполнено уже при λ = 0), и «не сошлось» (шаги кончились или упёрлось
в потолок шага).
"""
function iterate_need(need::Float64, s::Int; start::Float64 = START)
    lam = start
    path = [lam]
    for _ in 1:STEPS
        g = at(need, s, lam).pred
        (!isfinite(g) || g > CEIL) && return (path = path, lam = Inf, how = :потолок)
        push!(path, g)
        abs(g - lam) < TOL && return (path = path, lam = g, how = :сошлось)
        g <= 0.0 && return (path = path, lam = 0.0, how = :без_строя)
        lam = g
    end
    return (path = path, lam = lam, how = :шагов)
end

"""
    ladder_star(need, s)

Наименьшее λ из лестницы, при котором промах этого зерна не хуже требования, —
независимое от формулы чтение: то, что измерено поведением, а не равновесием.
"""
function ladder_star(need::Float64, s::Int; detune::Float64 = 0.0,
                     ladder = LADDER)
    for lam in ladder
        at(need, s, lam; detune = detune).miss <= need && return lam
    end
    return NaN
end

"""
    iterate_verdict(need, seed; start, detune, steps)

Та же итерация, но исход называется точнее: «точка» — сошлось; «цикл2» — двухтактный
цикл (g(g(λ)) вернулось к λ, а g(λ) — нет: неподвижной точки у среды нет, только
пара); «без строя» — предел не выше нуля; «шагов» — не дошло. Различение важно:
осцилляция и остановка посреди спуска дают разные числа, и путать их нельзя.
"""
function iterate_verdict(need::Float64, s::Int; start::Float64 = START,
                         detune::Float64 = 0.0, steps::Int = 60)
    lam = start
    path = [lam]
    for _ in 1:steps
        g = at(need, s, lam; detune = detune).pred
        isfinite(g) || return (path = path, lam = NaN, how = :срыв)
        push!(path, g)
        abs(g - lam) < TOL && return (path = path, lam = g, how = :точка)
        h = at(need, s, g; detune = detune).pred
        isfinite(h) && abs(h - lam) < TOL &&
            return (path = path, lam = (lam + g) / 2, how = :цикл2)
        g <= 0.0 && return (path = path, lam = 0.0, how = :без_строя)
        lam = g
    end
    return (path = path, lam = lam, how = :шагов)
end

"Исход итерации к печати."
function verdict_txt(how::Symbol)
    how == :точка && return "точка"
    how == :цикл2 && return "двухтактный цикл"
    how == :без_строя && return "без строя"
    how == :срыв && return "сорвалась"
    return "не дошло"
end

"Числа к печати размахом: наименьшее … наибольшее; «не нашлось» — тоже ответ."
function span_txt(values)
    good = filter(isfinite, values)
    isempty(good) && return "не нашлось"
    return @sprintf("%7.2f … %-7.2f", minimum(good), maximum(good))
end

"Как сошлось — к печати."
function how_txt(how::Symbol)
    how == :сошлось && return "сошлось"
    how == :без_строя && return "без строя"
    how == :потолок && return "потолок"
    return "не сошлось"
end

# --- чтение ---------------------------------------------------------------

"Средняя ошибка ряда против лестницы: |v − l|/l по тем ячейкам, где оба числа есть."
function mean_err(vals, ref)
    xs = [abs(v - l) / l for (v, l) in zip(vals, ref) if isfinite(v) && isfinite(l) && l > 0]
    return isempty(xs) ? NaN : sum(xs) / length(xs)
end

"g(λ) по шести зёрнам — предсказание, снятое при этой жёсткости."
g_values(need::Float64, lam::Float64) = [at(need, s, lam).pred for s in SEEDS]

"Сколько зёрен из шести дали промах не хуже требования при этой жёсткости."
held_count(need::Float64, lam::Float64; detune::Float64 = 0.0) =
    count(s -> at(need, s, lam; detune = detune).miss <= need, SEEDS)

function main_base()
    println("λ — не производная, а корень: нужная жёсткость при своём λ.")
    println("Среда та же, что в `tension.jl`: три моды, связь 0.6, полоса 0.1, затухание")
    println("0.8, память 1e-3. Шесть зёрен, шаг описания 0.01; потолок шага — 199.2.")
    println("Формула читается как неравенство λ ≥ g(λ): её решением будет множество")
    println("держащих жёсткостей, а нужная — наименьшее в нём. g снимается при своём λ.")
    @printf("Чужие точки снятия — те же, что в `tension.jl`: %.0f и %.0f.\n", START, FAR)

    println("\n--- самосогласование: итерация λ ← g(λ) " * "-"^6)
    println("   Итерация от чужой точки 16 сходит к своему λ. Печатается размах предела по")
    println("   зёрнам, путь у двух зёрен, шаги и чем кончилось; ниже — от других начал.")
    iters = Dict{Tuple{Float64,Float64}, Any}()
    for need in NEEDS, start in (1.0, START, 128.0)
        iters[(need, start)] = [iterate_need(need, s; start = start) for s in SEEDS]
    end
    for need in NEEDS
        res = iters[(need, START)]
        fins = [r.lam for r in res]
        hows = [r.how for r in res]
        @printf("\n   требование %4.0f %%   предел %s   %s\n", 100 * need,
                span_txt(fins), join(sort(unique([how_txt(h) for h in hows])), "/"))
        for (k, r) in enumerate(res)
            k > 2 && break
            path = r.path[1:min(end, 5)]
            @printf("      зерно %d: %s%s   шагов %d\n", k,
                    join([@sprintf("%.2f", v) for v in path], " → "),
                    length(r.path) > 5 ? " → …" : "", length(r.path) - 1)
        end
        fin = filter(isfinite, fins)
        @printf("      из шести: предел есть у %d, не потолком кончилось у %d\n",
                length(fin), count(h -> h != :потолок && h != :шагов, hows))
        @printf("      выше потолка %.1f: %s\n", CEIL,
                isempty(fin) ? "—" : (maximum(fin) > CEIL ? "да" : "нет"))
    end

    println("\n   Единственность: итерация от 1 и от 128 против той же, от 16.")
    for need in NEEDS
        base = [r.lam for r in iters[(need, START)]]
        for start in (1.0, 128.0)
            other = [r.lam for r in iters[(need, start)]]
            pairs = [(b, o) for (b, o) in zip(base, other) if isfinite(b) && isfinite(o)]
            worst = isempty(pairs) ? NaN :
                maximum(abs(b - o) / max(b, 1e-30) for (b, o) in pairs)
            @printf("   требование %4.0f %%   от %5.0f %s   расхождение самое большое %.2f %%\n",
                    100 * need, start, span_txt(other), 100 * worst)
        end
    end




    println("\n--- три числа рядом: своё λ, чужое λ и лестница " * "-"^2)
    println("   λ* — предел итерации при своём λ; g(16) и g(48) — предсказание при чужом, как")
    println("   в `tension.jl`; «лестница» — наименьшее λ из перебора, при котором промах этого")
    println("   зерна не хуже требования: измеренное поведением, а не равновесием. Печатается")
    println("   размах по зёрнам и — против лестницы — средняя ошибка.")
    for need in NEEDS
        mine = [r.lam for r in iters[(need, START)]]
        g16 = g_values(need, START)
        g48 = g_values(need, FAR)
        lad = [ladder_star(need, s) for s in SEEDS]
        @printf("\n   требование %4.0f %%\n", 100 * need)
        for (tag, vals) in (("своё λ*", mine), ("g(16)", g16), ("g(48)", g48),
                            ("лестница", lad))
            e = mean_err(vals, lad)
            @printf("      %-9s %s   (ошибка против лестницы %s)\n", tag, span_txt(vals),
                    isfinite(e) ? @sprintf("%5.0f %%", 100 * e) : "не нашлась")
        end
    end

    println("\n--- куда врёт формула: λ против g(λ) при держании и без " * "-"^2)
    println("   Формула говорит «держится», когда λ ≥ g(λ). Поведение говорит то же, когда")
    println("   промах не хуже требования. Считаются все ячейки лестницы: 17 ступеней × 6 зёрен")
    println("   на каждое требование.")
    verdicts = Dict{Float64,Any}()
    for need in NEEDS
        hold_n, says_n, hold_ok, hold_deep, fail_ok, fail_lax = 0, 0, 0, 0, 0, 0
        d_hold, d_fail = Float64[], Float64[]
        for lam in LADDER, s in SEEDS
            r = at(need, s, lam)
            held = r.miss <= need
            says = isfinite(r.pred) && lam >= r.pred
            held && (hold_n += 1)
            says && (says_n += 1)
            if held
                push!(d_hold, r.pred - lam)
                says ? (hold_ok += 1) : (hold_deep += 1)
            else
                push!(d_fail, r.pred - lam)
                says ? (fail_ok += 1) : (fail_lax += 1)
            end
        end
        span(v) = isempty(v) ? "—" : @sprintf("%+.3f … %+.3f", minimum(v), maximum(v))
        @printf("\n   требование %4.0f %%: держится %d ячеек, формула говорит «держится» %d\n",
                100 * need, hold_n, says_n)
        @printf("      держится: формула согласна %d, требует больше %d (пессимистична)\n",
                hold_ok, hold_deep)
        @printf("      не держится: формула согласна %d, разрешает зря %d (оптимистична)\n",
                fail_ok, fail_lax)
        @printf("      запас g(λ) − λ: где держится %s, где нет %s\n",
                span(d_hold), span(d_fail))
        verdicts[need] = (hold_n = hold_n, hold_ok = hold_ok, fail_lax = fail_lax)
    end


    println("\n--- пороги, названные до прогона " * "-"^7)
    fins, above = 0, 0
    for need in NEEDS, r in iters[(need, START)]
        if isfinite(r.lam)
            fins += 1
            r.lam > CEIL && (above += 1)
        end
    end
    @printf("   1. существование: предел есть у %d из 18 (зерно × требование), выше потолка %d\n",
            fins, above)
    uniq = 0.0
    for need in NEEDS
        base = [r.lam for r in iters[(need, START)]]
        for start in (1.0, 128.0)
            for (b, o) in zip(base, [r.lam for r in iters[(need, start)]])
                (isfinite(b) && isfinite(o)) || continue
                uniq = max(uniq, abs(b - o) / max(b, 1e-30))
            end
        end
    end
    @printf("   2. единственность: от 1 и от 128 против 16 — расхождение %.4f %% (%s)\n",
            100 * uniq, uniq < 0.01 ? "к одному числу" : "чисел больше одного")
    close_n, total = 0, 0
    for need in NEEDS
        lad = [ladder_star(need, s) for s in SEEDS]
        for (v, l) in zip([r.lam for r in iters[(need, START)]], lad)
            (isfinite(v) && isfinite(l) && l > 0) || continue
            total += 1
            abs(v - l) / l <= 0.25 && (close_n += 1)
        end
    end
    @printf("   3. совпадение с держанием: в 25 %% уложилось %d из %d ячеек (%s)\n",
            close_n, total, close_n >= total - 3 ? "ведёт туда же" : "ведёт не туда")
    for need in NEEDS
        mine = [r.lam for r in iters[(need, START)]]
        lad = [ladder_star(need, s) for s in SEEDS]
        e_mine, e_16 = mean_err(mine, lad), mean_err(g_values(need, START), lad)
        @printf("   4. чужое λ (требование %4.0f %%): своё λ* ошибается на %s, g(16) — на %s (%s)\n",
                100 * need, isfinite(e_mine) ? @sprintf("%.0f %%", 100 * e_mine) : "—",
                isfinite(e_16) ? @sprintf("%.0f %%", 100 * e_16) : "—",
                isfinite(e_16) && isfinite(e_mine) && e_mine <= e_16 ? "своё ближе" :
                "своё не ближе")
    end
    for need in NEEDS
        v = verdicts[need]
        @printf("   5. куда врёт (требование %4.0f %%): держащих ячеек %d, формула согласна с %d, разрешает зря %d\n",
                100 * need, v.hold_n, v.hold_ok, v.fail_lax)
    end
    println("   Итог — в журнале, а не «потом».")
    return nothing
end

"""
    main_detuned()

Второе чтение — на невырожденной мере. Аккорд сбит на 20 %, требование 1 % строго
ниже написанного: держаться значит сжимать. Вопрос тот же — есть ли у итерации
предел, один ли он, совпадает ли с измеренным поведением, — но мерить его теперь
нечем подменить.
"""
function main_detuned()
    println("\n=== второе чтение: невырожденная мера " * "="^8)
    println("   Аккорд сбит на 20 % (как в `tension.jl`), требование 1 % строго ниже")
    println("   написанного. Лестница от 4 до 96; итерация от 1 и от 16, исход называется")
    println("   точкой, двухтактным циклом или недосчётом.")
    need = 0.01
    ok_point, ok_unam, ok_far, n_lad = 0, 0, 0, 0
    for s in SEEDS
        lad, lad_miss = NaN, Inf
        for l in FINE
            r = at(need, s, l; detune = DETUNE)
            if r.miss <= need
                lad, lad_miss = l, r.miss
                break
            end
        end
        a = iterate_verdict(need, s; start = START, detune = DETUNE)
        b = iterate_verdict(need, s; start = 1.0, detune = DETUNE)
        same = isfinite(a.lam) && isfinite(b.lam) &&
               abs(a.lam - b.lam) / max(a.lam, 1e-30) < 0.01
        near = isfinite(a.lam) && isfinite(lad) && abs(a.lam - lad) / lad <= 0.25
        a.how == :точка && (ok_point += 1)
        same && (ok_unam += 1)
        near && (ok_far += 1)
        isfinite(lad) && (n_lad += 1)
        @printf("   зерно %d   лестница %8s (промах %s)   от 16: %-16s %.2f   от 1: %-16s %.2f\n",
                s, isfinite(lad) ? @sprintf("%.0f", lad) : "нет",
                isfinite(lad_miss) ? @sprintf("%.4f", lad_miss) : "—",
                verdict_txt(a.how), a.lam, verdict_txt(b.how), b.lam)
        path = a.path[1:min(end, 6)]
        @printf("      путь от 16: %s%s\n",
                join([@sprintf("%.2f", v) for v in path], " → "),
                length(a.path) > 6 ? " → …" : "")
    end
    println("\n   Порог 6, названный до этого прогона:")
    @printf("      уложились в 25 %% от лестницы: %d из %d ячеек\n", ok_far, n_lad)
    @printf("      сходились от разных начал к одному числу: %d из %d зёрен\n", ok_unam, 6)
    @printf("      были точкой (а не циклом и не недосчётом): %d из %d зёрен\n", ok_point, 6)
    if ok_far >= 5 && ok_unam >= 5 && ok_point >= 5
        println("      самосогласование живёт: причина первого чтения — вырожденная мера.")
    else
        println("      самосогласования нет и здесь: причина не в мере, а в равновесии.")
    end
    return nothing
end

"""
    main_still()

Третье чтение — о том, чем мерено. Второе чтение на сбитом аккорде дало: точка у
итерации есть и одна, но она выше лестничного порога. Две причины возможны, и они
разные: либо равновесие требует больше, чем нужно на деле (формула пессимистична),
либо за минуту состояние не успевает дойти до равновесия, и «держится» — про
минуту, а не про равновесие. Различить можно по промаху: если он между 60 и 120
секундами ещё идёт, держание временное; если стоит — установившееся.

Порог объявлен до этого прогона: держание считается установившимся, когда промах
не растёт во времени (через 120 с не хуже, чем через 60) и за последнюю секунду
сдвинулся меньше чем на 1e-5 — в обоих случаях при той жёсткости, где лестница
показала держание.
"""
function main_still()
    println("\n=== третье чтение: установилось ли держание " * "="^3)
    println("   Сбитый на 20 % аккорд, требование 1 %. Для каждой жёсткости, где промах")
    println("   уложился, смотрится тот же промах через две минуты и сдвиг за последнюю")
    println("   секунду: стоит промах или ещё идёт.")
    need = 0.01
    settled, walking = 0, 0
    for s in SEEDS
        lad = NaN
        for l in FINE
            at(need, s, l; detune = DETUNE).miss <= need && (lad = l; break)
        end
        isfinite(lad) || continue
        for lam in (lad, lad + 8.0)
            m60 = miss_at(s, lam; seconds = 60.0)
            m61 = miss_at(s, lam; seconds = 61.0)
            m120 = miss_at(s, lam; seconds = 120.0)
            m121 = miss_at(s, lam; seconds = 121.0)
            still = isfinite(m120) && m120 <= max(m60, need) && abs(m121 - m120) < 1e-5
            lam == lad && (still ? (settled += 1) : (walking += 1))
            @printf("   зерно %d   λ %5.1f   промах 60 с %.5f, 120 с %.5f   ход за секунду %.2e и %.2e   %s\n",
                    s, lam, m60, m120, abs(m61 - m60), abs(m121 - m120),
                    still ? "стоит" : "идёт")
        end
    end
    @printf("\n   Порог, названный до прогона: из %d держаний лестницы установились %d, шли %d\n",
            settled + walking, settled, walking)
    if settled >= 5
        println("      держание установившееся: расхождение с корнем — о формуле, не о минуте.")
    else
        println("      держание ещё шло: лестница меряет минуту, формула — равновесие.")
    end
    return nothing
end

"Промах через заданное время при данной жёсткости — прогон ради одного числа."
function miss_at(s::Int, lam::Float64; seconds::Float64, detune::Float64 = 0.0)
    m = tensioned(seed = s, tuning = lam, detune = detune)
    for _ in 1:round(Int, seconds / DT)
        step!(m, DT)
        (all(isfinite, m.omega) && all(isfinite, m.theta)) || return Inf
    end
    return x_over(m)
end

function main(which::Symbol)
    which in (:all, :base) && main_base()
    which in (:all, :detuned) && main_detuned()
    which in (:all, :still) && main_still()
    return nothing
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? :all : Symbol(ARGS[1]))
end
