# Entrain. Сходится ли описание.
#
# Это не опыт и не проверка мира. Утверждения описания сверяются друг с другом:
# то, что следует из уравнений, сравнивается с тем, что из них считается.
# Порогов здесь нет: есть точность чисел с плавающей точкой и точность схемы.
#
# Утверждение не должно зависеть от того, какой поток случайных чисел дала
# версия Julia: у 1.10 и 1.13 он разный. Поэтому всё, что зависит от броска,
# сверяется на нескольких начальных условиях сразу, а числа поведения
# печатаются отдельно — они к версии привязаны, и это не порок: измерение и
# должно быть привязано к тому, чем измерено.
#
# Сцены открытых вопросов (`modes.jl`, `boundary.jl`, `seam.jl`, `scales.jl`, `relief.jl`, `terms.jl`) читают;
# здесь то, что из этих чтений следует из уравнений, превращено в утверждения — то же
# число, но без броска генератора. Числа о среде, которые утверждением быть не могут
# (разброс по начальным условиям), печатаются отдельно, размахом и в трёх средах:
# свежей, прожившей без событий и прожившей с событием.
#
# Запуск:
#
#     julia julia/kuramoto/consistency.jl

include(joinpath(@__DIR__, "basic.jl"))

const TOL_EXACT = 1e-9     # численная сумма: сходится к нулю
const TOL_SCHEME = 5e-3    # точность схемы первого порядка
const WRITTEN = 1e-2       # написанный процент по имени: то, чем среда написана

"""
    report_zeros(name, error, tol)

Утверждение вида «величина равна нулю» (или «совпадает с формулой»).
"""
function report_zeros(name::AbstractString, error::Float64, tol::Float64)
    ok = error <= tol
    @printf("  %-56s %-11s %.2e\n", name, ok ? "сошлось" : "НЕ СОШЛОСЬ", error)
    return ok
end

"""
    report_at_least(name, value, least)

Утверждение вида «величина не меньше»: след есть, деформация произошла.
"""
function report_at_least(name::AbstractString, value::Float64, least::Float64)
    ok = value >= least
    @printf("  %-56s %-11s %.3e\n", name, ok ? "сошлось" : "НЕ СОШЛОСЬ", value)
    return ok
end

"""
    zero_coupling(; damping, shift, seed)

Среда без связи: ни одна вибрация не влияет на другие. Нужна, чтобы увидеть
затухание и свободное движение в чистом виде.
"""
function zero_coupling(; damping::Float64 = 0.0, shift::Float64 = 0.0, seed::Int = 4)
    m = build(memory = 0.0, damping = damping, seed = seed)
    m.coupling .= 0.0
    shift == 0.0 || perturb!(m, 1, shift)
    return m
end

"""
    listen_to(medium, indices, t)

То, что слышно в точке, если слушать только часть среды.
"""
listen_to(m::Medium, indices, t::Float64) = mean(cos.(m.omega[indices, :] .* t .+ m.theta[indices, :]))

"""
    response(medium; seconds, shift, step)

Отклик на одно и то же событие, в числах:
`depth`     — насколько ушла частота сразу после толчка;
`residual`  — где частота остаётся потом (среда стоит не на своей, а на своей в среде);
`half_time` — за сколько секунд прошла половина пути (скорость отклика);
`shape`     — насколько возврат отличается от простого затухания к остатку:
              ноль означал бы ровно экспоненциальный возврат.

Это измерение поведения, а не утверждение.
"""
function response(m::Medium; seconds::Float64 = 6.0, shift::Float64 = 1.5,
                    step::Float64 = 0.05)
    perturb!(m, 1, shift)
    depth = drift(m, 1)
    times, devs = Float64[], Float64[]
    t = 0.0
    while t < seconds - 1e-9
        advance!(m, step)
        t += step
        push!(times, t)
        push!(devs, drift(m, 1))
    end
    residual = devs[end]
    k = findfirst(d -> d <= residual + (depth - residual) / 2, devs)
    half_time = k === nothing ? NaN : times[k]
    plain = residual .+ (depth - residual) .* exp.(-log(2) .* times ./ half_time)
    shape = maximum(abs.(devs .- plain) ./ (depth - residual))
    return (depth = depth, residual = residual, half_time = half_time, shape = shape)
end

"""
    echo(medium, i = 5; shift, seconds)

Насколько событие дошло до вибрации i: среда проживается дважды — с событием
и без него, — и сравнивается одно и то же место. Толчок в одном месте
отзывается в другом.
"""
function echo(m::Medium, i::Int = 5; shift::Float64 = 1.5, seconds::Float64 = 6.0)
    quiet = deepcopy(m)
    perturb!(m, 1, shift)
    advance!(m, seconds)
    advance!(quiet, seconds)
    return abs(m.omega[i, 1] - quiet.omega[i, 1])
end

"""
    link_goal(medium, i, j)

То, к чему идёт связь вибраций i и j: Σₚₙ S(ωⱼₙ−ωᵢₚ)·cos(θᵢₚ−θⱼₙ) / Σₚₙ S.
Это ровно та цель, к которой связь сдвигается за шаг, — и по ней читается,
какой долей пути она идёт.
"""
function link_goal(m::Medium, i::Int, j::Int)
    acc, weight = 0.0, 0.0
    for p in axes(m.omega, 2), n in axes(m.omega, 2)
        s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
        acc += s * cos(m.theta[i, p] - m.theta[j, n])
        weight += s
    end
    return weight == 0.0 ? m.coupling[i, j] : acc / weight
end

"""
    pull(medium)

Суммарная тяга среды самой себя:
Σᵢⱼₚₙ Wᵢⱼ S(ωⱼₙ−ωᵢₚ) sin(θⱼₙ−θᵢₚ).
При симметричной связи равна нулю: среда не может сдвинуть себя целиком.
"""
function pull(m::Medium)
    total = 0.0
    vibs, modes = size(m.omega)
    for i in 1:vibs, j in 1:vibs
        i == j && continue
        for p in 1:modes, n in 1:modes
            s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
            total += m.coupling[i, j] * s * sin(m.theta[j, n] - m.theta[i, p])
        end
    end
    return total
end

"""
    drag(medium, i, p)

Тяга связи на моду (i, p) — ровно тот член, что складывается в `step!`, без строя
и без затухания:

Σⱼₙ Wᵢⱼ S(ωⱼₙ−ωᵢₚ) sin(θⱼₙ−θᵢₚ).

Со строем и затуханием он и спорит: промах обертона стоит там, где силы
уравновесились.
"""
function drag(m::Medium, i::Int, p::Int)
    d = 0.0
    for j in 1:size(m.omega, 1)
        j == i && continue
        for n in 1:size(m.omega, 2)
            s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
            d += m.coupling[i, j] * s * sin(m.theta[j, n] - m.theta[i, p])
        end
    end
    return d
end

"""
    state_gap(a, b)

Отличие двух состояний: наибольшее из отличий по частотам и по фазам. Ноль
значит, что числа одни и те же, — не «близки», а те же.
"""
function state_gap(a::Medium, b::Medium)
    return max(maximum(abs.(a.omega .- b.omega)), maximum(abs.(a.theta .- b.theta)))
end

"""
    chain(; reach, tie, gap, band, spread, seed, one_sided)

Среда-цепочка из двенадцати вибраций и гость — тринадцатая: сцена `boundary.jl`
без прогрева и с выключенным затуханием. Связь есть только с ближними на
расстоянии не больше `reach`, поэтому у края её вдвое меньше; гость пришит к
краю одной нитью силы `tie` и отстоит своей частотой на `gap`.

Односторонняя нить (`one_sided`) — не среда, а ошибка в среде: она нужна здесь
только затем, чтобы показать, что тяга есть и она не ноль.
"""
function chain(; reach::Int = 3, tie::Float64 = 0.0, gap::Float64 = 0.0,
               band::Float64 = Inf, spread::Float64 = 0.03, seed::Int = 1,
               one_sided::Bool = false)
    n, t = 12, 13
    rng = MersenneTwister(seed)
    omega0 = zeros(t, 1)
    omega0[1:n, 1] = 1.0 .+ spread .* randn(rng, n)
    omega0[t, 1] = mean(omega0[1:n, 1]) + gap
    theta = 2π .* rand(rng, t, 1)
    W = zeros(t, t)
    for i in 1:n, j in 1:n
        d = abs(i - j)
        (d == 0 || d > reach) && continue
        W[i, j] = 0.6 / (2 * reach)
    end
    W[t, n] = tie
    one_sided || (W[n, t] = tie)
    return Medium(omega0 = omega0, theta = theta, coupling = W, band = band, damping = 0.0)
end

"""
    full_link(link, n)

Связь, полная между всеми: у каждой пары `link`, себя никто не тянет.
"""
function full_link(link::Float64, n::Int)
    W = fill(link, n, n)
    for i in 1:n
        W[i, i] = 0.0
    end
    return W
end

"""
    lump_pair(; tie, gap, band, spread, seed)

Два сгустка по шесть вибраций и одна перемычка между ними: сцена `seam.jl` без
прогрева и с выключенным затуханием. Внутри сгустка связь полная — 0.6/6 на
пару; перемычка силы `tie` — одна связь 6-7.

Перемычка, равная внутренней связи, задаётся тем же числом (0.6/6), а не
похожим (0.1): 0.6/6 и 0.1 отличаются на бит, и за минуту это расходится до
целых чисел — сравнивались бы не среды, а последний бит сложения.
"""
function lump_pair(; tie::Float64 = 0.0, gap::Float64 = 0.0, band::Float64 = Inf,
                   spread::Float64 = 0.01, seed::Int = 1)
    n, t = 6, 12
    rng = MersenneTwister(seed)
    omega0 = zeros(t, 1)
    omega0[1:n, 1] .= 1.0 .+ spread .* randn(rng, n)
    omega0[n+1:t, 1] .= 1.0 .+ gap .+ spread .* randn(rng, n)
    theta = 2π .* rand(rng, t, 1)
    W = zeros(t, t)
    for i in 1:n, j in 1:n
        i == j && continue
        W[i, j] = 0.6 / n
        W[n+i, n+j] = 0.6 / n
    end
    W[n, n+1] = tie
    W[n+1, n] = tie
    return Medium(omega0 = omega0, theta = theta, coupling = W, band = band, damping = 0.0)
end

"""
    checks()

Все утверждения описания, которые можно сверить числами.
"""
function checks()
    results = Bool[]

    println("связь и состояние")

    m = build()
    push!(results, report_zeros("Σᵢⱼ Wᵢⱼ sin(θⱼ−θᵢ) = 0 при симметричной связи", abs(pull(m)), TOL_EXACT))

    m = build(modes = 3, band = 0.1)
    push!(results, report_zeros("то же с аккордом и полосой (резонанс не выдумывает движение)", abs(pull(m)), TOL_EXACT))

    m = build(damping = 0.0)
    before = sum(m.omega)
    advance!(m, 20.0)
    push!(results, report_zeros("сумма частот сохраняется (γ = 0)", abs(sum(m.omega) - before), TOL_EXACT))

    m = build(modes = 3, band = 0.1, damping = 0.0)
    before = sum(m.omega)
    advance!(m, 20.0)
    push!(results, report_zeros("сумма частот всех мод сохраняется (γ = 0)", abs(sum(m.omega) - before), TOL_EXACT))

    println("\nсвободное движение и затухание")

    m = zero_coupling()
    theta0, omega0 = copy(m.theta), copy(m.omega)
    advance!(m, 10.0)
    free = maximum(abs.(m.theta .- (theta0 .+ omega0 .* 10.0)))
    push!(results, report_zeros("без связи θ(t) = θ(0) + ω t", free, TOL_EXACT))

    m = zero_coupling(damping = 0.7, shift = 1.5)
    d0 = drift(m, 1)
    advance!(m, 4.0)
    rate = log(d0 / drift(m, 1)) / 4.0
    push!(results, report_zeros("скорость затухания равна γ", abs(rate - 0.7) / 0.7, TOL_SCHEME))

    println("\nпамять")

    never = build(memory = 0.0)
    aged = build(memory = 0.0)
    for _ in 1:200
        perturb!(aged, 1, 1.5)
        advance!(aged, 6.0)
    end
    push!(
        results,
        report_zeros(
            "при ε = 0 событие не меняет среду",
            maximum(abs.(aged.coupling .- never.coupling)),
            TOL_EXACT,
        ),
    )

    never = build(memory = 1e-3)
    aged = build(memory = 1e-3)
    for _ in 1:200
        perturb!(aged, 1, 1.5)
        advance!(aged, 6.0)
    end
    push!(
        results,
        report_at_least(
            "при ε > 0 событие меняет среду (след есть)",
            maximum(abs.(aged.coupling .- never.coupling)),
            0.01,
        ),
    )

    println("\nмасштаб памяти: один ε или много")

    # Масштаб виден по шагу: связь идёт к своей цели долей ε·dt, и доля эта одна
    # у всех связей уровня. В одной среде два уровня, и у каждого своя доля: у
    # связей вибраций ε, у связей мод внутри — εᵤ. Сверяются обе доли сразу, на
    # одном шаге одной среды, — иначе «свой масштаб у уровня» ничем не отличалось
    # бы от «одно ε на всё».
    worst, share_spread = 0.0, 0.0
    for seed in 1:6
        m = build(modes = 3, band = 0.1, inner = 0.5, inner_memory = 1e-4,
                  memory = 3e-3, seed = seed)
        was = copy(m.coupling)
        inside = [copy(world(m, i).coupling) for i in 1:N]
        step!(m, DT)
        outer = Float64[]
        for i in 1:N, j in 1:N
            i == j && continue
            d = link_goal(m, i, j) - was[i, j]
            abs(d) < 1e-12 && continue
            push!(outer, (m.coupling[i, j] - was[i, j]) / d)
        end
        inner = Float64[]
        for i in 1:N, p in 1:3, q in 1:3
            p == q && continue
            d = cos(m.theta[i, p] - m.theta[i, q]) - inside[i][p, q]
            abs(d) < 1e-12 && continue
            push!(inner, (world(m, i).coupling[p, q] - inside[i][p, q]) / d)
        end
        worst = max(worst, maximum(abs.(outer .- DT * 3e-3)),
                           maximum(abs.(inner .- DT * 1e-4)))
        share_spread = max(share_spread, maximum(outer) - minimum(outer),
                                         maximum(inner) - minimum(inner))
    end
    push!(results, report_zeros("за шаг связь идёт к своей цели долей ε·dt: у вибраций ε, у мод внутри εᵤ",
                                worst, TOL_EXACT))
    push!(results, report_zeros("и доля одна у всех связей уровня: масштаб у уровня один, не спектр",
                                share_spread, TOL_EXACT))
    # Это — пока ε у уровня одно. Раскладка ε по связям (глава 22, «срок у связи»)
    # ниже: там доля у каждой связи своя, а при одном ε у всех запись прежняя.

    # Прожитое пишется, только если его хватило на срок памяти: наводит связь
    # произведение ε·прожитое, а не событие. При ε·T = 0.012 связь остаётся при
    # своей (0.070 против начальной 0.060), при 1.2 — уходит больше половины
    # пути (0.707…0.710; цель этой среды, её собственная согласованность, ≈ 0.70,
    # так что путь — 0.64, а половина — 0.32). Время здесь одно и то же — 1200 с:
    # меняется только ε, то есть срок памяти 1/ε.
    worst, least = 0.0, Inf
    for seed in 1:6
        little = build(memory = 1e-5, seed = seed)
        plenty = build(memory = 1e-3, seed = seed)
        for _ in 1:200
            advance!(little, 6.0)
            advance!(plenty, 6.0)
        end
        worst = max(worst, abs(mean_link(little) - 0.6 / N))
        least = min(least, mean_link(plenty) - 0.6 / N)
    end
    push!(results, report_zeros("пока ε·T = 0.012, связь остаётся при своей: мало прожитого — мало написано",
                                worst, 2e-2))
    push!(results, report_at_least("а при ε·T = 1.2 связь идёт больше половины пути: срок памяти — это 1/ε",
                                   least, 0.3))

    # Окно 1/ε решает и что останется в записи: долгая память пишет среднюю по
    # прожитому, и она у всех сред одна (разброс 0.002…0.003), короткая пишет то,
    # что сейчас, и оно у каждой своё (0.2…0.7). Это не «числа разные»: у одного
    # и того же прожитого срока разные ε дают разные по природе записи.
    widths = Float64[]
    for ε in (1e-3, 1e-1)
        v = Float64[]
        for seed in 1:6
            m = build(memory = ε, seed = seed)
            for _ in 1:200
                advance!(m, 6.0)
            end
            push!(v, mean_link(m))
        end
        push!(widths, maximum(v) - minimum(v))
    end
    push!(results, report_zeros("долгая память (1/ε = 1000 с) пишет одно на всех: разброс по средам мал",
                                widths[1], 1e-2))
    push!(results, report_at_least("а короткая (10 с) пишет то, что сейчас: разброс у каждой свой",
                                   widths[2], 5e-2))

    # Событие в широкую запись не входит вовсе: среда, через которую событие шло
    # двести раз, и среда того же зерна и возраста без событий накопили одну и ту
    # же связь (разница 0.006). Значит пишется прожитое, а событие входит в запись
    # только через состояние — там, где окно короче прожитого.
    worst = 0.0
    for seed in 1:6
        routed = build(memory = 1e-3, seed = seed)
        quiet = build(memory = 1e-3, seed = seed)
        for _ in 1:200
            perturb!(routed, 1, 1.5)
            advance!(routed, 6.0)
            advance!(quiet, 6.0)
        end
        worst = max(worst, abs(mean_link(routed) - mean_link(quiet)))
    end
    push!(results, report_zeros("широкая память события не различает: с событием и без него связь одна",
                                worst, 2e-2))

    # И устойчивость даёт тоже прожитое: у прожившей среды отзвук одинаков по
    # начальным условиям (разброс 0.0002…0.0006), у свежей гуляет (0.2 и больше).
    # Утверждение не о величине отклика, а о его воспроизводимости.
    fresh_echo, lived_echo = Float64[], Float64[]
    for seed in 1:6
        f = build(memory = 1e-3, seed = seed)
        q = build(memory = 1e-3, seed = seed)
        for _ in 1:200
            perturb!(q, 1, 1.5)
            advance!(q, 6.0)
        end
        push!(fresh_echo, echo(f))
        push!(lived_echo, echo(q))
    end
    spent = maximum(fresh_echo) - minimum(fresh_echo)
    steadied = maximum(lived_echo) - minimum(lived_echo)
    push!(results, report_at_least("устойчивость даёт прожитое: разброс отзвука у прожившей в 10 раз уже",
                                   spent / max(steadied, 1e-12), 10.0))

    println("\nместо записи: что помнит связь, кроме времени")

    # Доля пути у всех связей уровня одна, а цель у каждой своя — потому из одной
    # доли выходит разная запись. Утверждений три, и они о разном: доля — одно
    # число на уровень, цели — разные у связей одного состояния, записанное —
    # разные у связей того же состояния. Без второго и третьего первое читалось бы
    # как «все связи пишут одинаково», а это неверно (сцена `relief.jl`).
    # Порог у третьей строки — пол, а не прочитанное: она говорит только, что
    # запись за шаг не одна, поэтому и порог низкий (измеренное в наименьшем зерне
    # 1.18, на другой версии 1.33).
    share_err, goal_spread, write_ratio = 0.0, Inf, Inf
    for seed in 1:6
        m = build(memory = 1e-3, seed = seed)
        was = copy(m.coupling)
        step!(m, DT)
        goals, written = Float64[], Float64[]
        for i in 1:N, j in 1:N
            i == j && continue
            push!(goals, link_goal(m, i, j))
            push!(written, abs(m.coupling[i, j] - was[i, j]))
            d = link_goal(m, i, j) - was[i, j]
            abs(d) < 1e-12 && continue
            share_err = max(share_err, abs((m.coupling[i, j] - was[i, j]) / d - DT * 1e-3))
        end
        goal_spread = min(goal_spread, maximum(goals) - minimum(goals))
        write_ratio = min(write_ratio, maximum(written) / median(written))
    end
    push!(results, report_zeros("за шаг связь идёт к цели долей ε·dt — и у всех связей доля одна",
                                share_err, TOL_EXACT))
    push!(results, report_at_least("а цели связей одного состояния разные: разброс целей не меньше",
                                   goal_spread, 1.5))
    push!(results, report_at_least("и записанное за шаг не одно: наибольшая связь пишет больше середины",
                                   write_ratio, 1.05))

    # Место записывается: связь, которой событие касается, отличается от связи
    # того же зерна и возраста, прожившей без событий, сильнее, чем связь вдали от
    # места. Проверяется на двух местах сразу: если бы рельеф стоял где придётся, у
    # второго места он стоял бы не там, где событие шло. Порог — не машинный и не
    # схемный: это измеренная величина, и у мест она разная (у 1-й вибрации 21.5 в
    # худшем зерне, у 5-й — 7.1: место записано и там, но слабее — сказано как
    # прочитано, сцены `relief.jl`).
    function lived(ε::Float64, passes::Int, seed::Int; place::Int = 0)
        m = build(memory = ε, seed = seed)
        for _ in 1:passes
            place != 0 && perturb!(m, place, 1.5)
            advance!(m, 6.0)
        end
        return m
    end
    place1, place5 = Inf, Inf
    wide_reliefs = Matrix{Float64}[]
    for seed in 1:6
        quiet = lived(1e-3, 200, seed)
        to1 = relief_of(lived(1e-3, 200, seed; place = 1), quiet)
        to5 = relief_of(lived(1e-3, 200, seed; place = 5), quiet)
        push!(wide_reliefs, to1)
        n1, f1 = relief_place(to1, 1)
        n5, f5 = relief_place(to5, 5)
        place1 = min(place1, n1 / f1)
        place5 = min(place5, n5 / f5)
    end
    push!(results, report_at_least("место записано: у 1-й вибрации отличие больше, чем вдали от неё",
                                   place1, 5.0))
    push!(results, report_at_least("и у 5-й тоже: событие шло там — там и рельеф",
                                   place5, 5.0))

    # Чей это рельеф: долгая память пишет среднее по прожитому, и рельеф у неё
    # один и тот же при любом начальном условии (похожесть 0.93…0.99), короткая
    # пишет то, что сейчас, и рельеф у каждой свой (у места события размер гуляет
    # в десятки раз). Тот же урок, что у средней связи, но про место, а не про
    # уровень.
    least_agree = Inf
    for a in 1:6, b in (a + 1):6
        least_agree = min(least_agree, relief_agreement(wide_reliefs[a], wide_reliefs[b]))
    end
    short_near = Float64[]
    for seed in 1:6
        quiet = lived(1e-2, 200, seed)
        δ = relief_of(lived(1e-2, 200, seed; place = 1), quiet)
        push!(short_near, relief_place(δ, 1)[1])
    end
    push!(results, report_at_least("долгая память пишет один рельеф на все условия: похожесть не меньше",
                                   least_agree, 0.5))
    push!(results, report_at_least("а короткая пишет рельеф свой: размер у места гуляет не меньше",
                                   maximum(short_near) / minimum(short_near), 10.0))

    # И тает рельеф по сроку памяти: за T секунд тишины — в e^(ε·T) раз. Проверяется
    # двумя памятями сразу (1/ε = 1000 с и 500 с): у каждой свой срок, и обе дают
    # ровно свой. Значит срок памяти читается и на стирании — тем же ε, которым
    # среда писала (сцена `relief.jl`; расхождение — десятые доли процента).
    worst_fade = 0.0
    for ε in (1e-3, 2e-3), seed in 1:3
        routed = lived(ε, 200, seed; place = 1)
        before = relief_place(relief_of(routed, lived(ε, 200, seed)), 1)[1]
        advance!(routed, 1200.0)
        after = relief_place(relief_of(routed, lived(ε, 400, seed)), 1)[1]
        worst_fade = max(worst_fade, abs(before / after / exp(ε * 1200.0) - 1))
    end
    push!(results, report_zeros("рельеф тает по сроку памяти: за 1200 с тишины — в e^(ε·1200) раз",
                                worst_fade, 2e-2))

    println("\nсрок у связи: ε не у уровня, а у каждой связи")

    # Шаг записи, названный в главе 22 дважды — в масштабах памяти (чтобы рядом
    # стояли мгновенное и годовое) и в месте (чтобы написанное пережило окно), —
    # стал записью: ε у связи (`memory_matrix`), а не у уровня. Утверждений шесть.
    #
    # Первое — что это не вторая запись. Раскладка ε с одним и тем же числом у всех
    # связей даёт те же числа, что ε уровня, — и не «почти», а ровно: в шаге то же
    # сложение, меняется только то, откуда взят множитель. Поэтому порог здесь не
    # машинный, а нуль: строка говорит «то же число», а не «близкое».
    wrong_split = 0.0
    for ε in (1e-5, 1e-3, 1e-2), seed in 1:2
        level = build(memory = ε, seed = seed)
        split = build(memory = ε, memory_matrix = fill(ε, N, N), seed = seed)
        advance!(level, 300.0)
        advance!(split, 300.0)
        wrong_split = max(wrong_split, maximum(abs.(level.omega .- split.omega)),
                          maximum(abs.(level.theta .- split.theta)),
                          maximum(abs.(level.coupling .- split.coupling)))
    end
    push!(results, report_zeros("ε одно у всех связей ⟹ раскладка по связям даёт те же числа",
                                wrong_split, 0.0))

    # Второе — то же на меньшем масштабе: мир внутри вибрации есть та же среда, и
    # раскладка у него устроена так же. Мир здесь с ненулевой связью (`inner ≠ 0`):
    # иначе в него нечему было бы войти и строка ничего не говорила бы.
    wrong_inner = 0.0
    for εᵤ in (1e-5, 1e-4), seed in 1:2
        level = build(modes = 3, band = 0.1, tuning = 8.0, inner = 0.5,
                      inner_memory = εᵤ, seed = seed)
        split = build(modes = 3, band = 0.1, tuning = 8.0, inner = 0.5,
                      inner_memory = εᵤ, inner_memory_matrix = fill(εᵤ, 3, 3), seed = seed)
        advance!(level, 300.0)
        advance!(split, 300.0)
        wrong_inner = max(wrong_inner, maximum(abs.(level.omega .- split.omega)),
                          maximum(abs.(level.theta .- split.theta)))
        for i in 1:N
            wrong_inner = max(wrong_inner, maximum(abs.(world(level, i).coupling .-
                                                        world(split, i).coupling)))
        end
    end
    push!(results, report_zeros("и у мира внутри вибрации: εᵤ одно у всех пар мод ⟹ те же числа",
                                wrong_inner, 0.0))

    # Третье — доля пути у связи своя и равна её ε·dt. Выше (у «доли пути») та же
    # доля сверялась как одна на уровень; здесь ε у связей одного состояния разные,
    # и берётся доля из самой раскладки, а не из числа уровня.
    split_share = 0.0
    for seed in 1:6
        ε = fill(1e-3, N, N)
        ε[1, 2], ε[1, 3] = 1e-2, 1e-4
        m = build(memory_matrix = ε, seed = seed)
        was = copy(m.coupling)
        step!(m, DT)
        for i in 1:N, j in 1:N
            i == j && continue
            d = link_goal(m, i, j) - was[i, j]
            abs(d) < 1e-12 && continue
            split_share = max(split_share,
                              abs((m.coupling[i, j] - was[i, j]) / d - DT * ε[i, j]))
        end
    end
    push!(results, report_zeros("за шаг связь идёт к цели долей своего ε·dt — у каждой связи своё",
                                split_share, TOL_EXACT))

    # Четвёртое — рядом в одной среде два срока: быстрая связь (ε = 1e-2, 1/ε = 100 с)
    # уходит к своей цели, медленная (ε = 1e-4, 1/ε = 10000 с) остаётся при своей.
    # Мера — накопленное за одно и то же прожитое время (1200 с) с одного состояния.
    # Порог объявлен до прогона и низкий нарочно: связь пишет долю пути, а не ε,
    # поэтому различается записанное не в сто раз (в сто раз различались бы доли).
    # В сцене `terms.jl` прочитано 8.9…9.6, в сверке по худшему из шести зёрен — 9.0:
    # быстрая связь прошла свою цель целиком (ε·T = 12), а медленная — восьмую часть
    # своей (ε·T = 0.12). Порог взят вдвое ниже прочитанного и говорит только, что
    # записанное у двух связей одного места разное.
    function with_terms(ε::Matrix{Float64}, passes::Int, seed::Int; place::Int = 0)
        m = build(memory_matrix = ε, seed = seed)
        for _ in 1:passes
            place != 0 && perturb!(m, place, 1.5)
            advance!(m, 6.0)
        end
        return m
    end
    terms_matrix() = (ε = fill(1e-3, N, N); ε[1, 2] = 1e-2; ε[1, 3] = 1e-4; ε)
    least_written = Inf
    for seed in 1:6
        ε = terms_matrix()
        keep = build(memory_matrix = ε, seed = seed)
        start = copy(keep.coupling)
        for _ in 1:200
            perturb!(keep, 1, 1.5)
            advance!(keep, 6.0)
        end
        least_written = min(least_written, abs(keep.coupling[1, 2] - start[1, 2]) /
                                            abs(keep.coupling[1, 3] - start[1, 3]))
    end
    push!(results, report_at_least("две связи одного места помнят разное: за то же время записано больше",
                                   least_written, 5.0))

    # Пятое и шестое — то, ради чего шаг назван при чтении места: написанное
    # переживает окно, если ε у связи. Место одно (первая вибрация), сроки у двух
    # его связей разные. Мера — отличие от среды того же зерна и возраста без
    # событий, до тишины и после 1200 с тишины. Ждётся e^(-ε·T): 0.887 у медленной
    # связи и 6.1e-6 у быстрой. Порогов два, оба объявлены до прогона. Первый —
    # половина написанного: в сцене `terms.jl` у медленной связи прочитано 0.887 у
    # всех шести зёрен, от теоретического e^(-0.12) = 0.887 отступ в третьем знаке.
    # Второй — «во много раз меньше»: остаток у быстрой связи гуляет по зёрнам от
    # 5.7e-6 до 4.5e-4, поэтому порог на него был бы не утверждением, а броском.
    # Держится он не сам: у одного места связь стоит не одна, и через общее
    # состояние остаток быстрой связи поддерживает медленная. Поэтому шестая
    # строка говорит не «остаток у быстрой мал», а «у медленной остаётся больше во
    # столько раз» — прочитано около двух тысяч, порог взят в десять.
    least_slow, least_gap = Inf, Inf
    for seed in 1:6
        ε = terms_matrix()
        routed = with_terms(ε, 200, seed; place = 1)
        quiet = with_terms(ε, 200, seed)
        before = relief_of(routed, quiet)
        advance!(routed, 1200.0)
        after = relief_of(routed, with_terms(ε, 400, seed))
        kept_slow = abs(after[1, 3]) / abs(before[1, 3])
        kept_fast = abs(after[1, 2]) / abs(before[1, 2])
        least_slow = min(least_slow, kept_slow)
        least_gap = min(least_gap, kept_slow / kept_fast)
    end
    push!(results, report_at_least("написанное переживает окно у медленной связи: остаётся не меньше",
                                   least_slow, 0.5))
    push!(results, report_at_least("а у быстрой уходит скорее: у медленной остаётся больше во столько раз",
                                   least_gap, 10.0))

    println("\nсогласованность и наложение")

    m = build()
    m.theta .= m.theta[1, 1]
    push!(results, report_zeros("все фазы равны ⟹ согласованность = 1", abs(coherence(m) - 1.0), TOL_EXACT))

    m.theta .= 2π .* (0:(N - 1)) ./ N
    push!(results, report_zeros("фазы разложены равномерно ⟹ согласованность = 0", coherence(m), TOL_EXACT))

    m = build()
    ρ = coherence(m)
    m.theta .+= 1.2345
    push!(results, report_zeros("общий сдвиг фаз не меняет согласованность", abs(coherence(m) - ρ), TOL_EXACT))

    m = build()
    # Разбиение — по числу вибраций среды, а не по десяти: при N = 10 это те же
    # массивы до бита, а при другом N утверждение остаётся тем же — целое есть
    # взвешенная сумма частей.
    part_a, part_b = [1, 2, 3], collect(4:N)
    whole = wave(m, 1.7)
    split = (length(part_a) * listen_to(m, part_a, 1.7) + length(part_b) * listen_to(m, part_b, 1.7)) / N
    push!(results, report_zeros("наложение: целое = взвешенная сумма частей", abs(whole - split), TOL_EXACT))

    println("\nаккорд: несколько частот у одной вибрации")

    m = build()
    push!(results, report_zeros("при одной моде ширина аккорда = 0", spread(m, 1), TOL_EXACT))

    m = build(modes = 3, band = 0.1)
    push!(results, report_at_least("три моды — аккорд шириной больше частоты", spread(m, 1), 1.0))
    advance!(m, 60.0)
    push!(results, report_at_least("через минуту аккорд всё ещё аккорд (моды не слились)", spread(m, 1), 1.0))

    Π = 0.1
    overshoot = maximum(max(overlap(Δ, Π) - (Π / Δ)^2, 0.0) for Δ in 0.2:0.05:20.0)
    push!(results, report_zeros("отклик падает не медленнее, чем (Π/Δ)²", overshoot, TOL_EXACT))

    # Строй тянет обертоны к их отношению. Утверждение — «держит лучше, чем
    # без него», и оно должно быть верно при любых начальных условиях, а не
    # при одном броске: генератор случайных чисел у разных версий Julia даёт
    # разные потоки, и утверждение не должно от этого зависеть. Поэтому
    # проверяем на нескольких начальных условиях, а число печатаем как
    # наихудшее из них — это уже измерение, не утверждение.
    worst = Inf
    seeds = 1:12
    for seed in seeds
        held = build(modes = 3, band = 0.1, tuning = 8.0, seed = seed)
        loose = build(modes = 3, band = 0.1, tuning = 0.0, seed = seed)
        advance!(held, 60.0)
        advance!(loose, 60.0)
        worst = min(worst, tuning_error(loose, 1) / (tuning_error(held, 1) + 1e-12))
    end
    push!(
        results,
        report_at_least(
            "строй держит обертоны лучше, чем без него (12 начальных условий)",
            worst,
            1.0 + TOL_EXACT,
        ),
    )

    m = build(modes = 3, band = 0.1, memory = 1e-3, tuning = 8.0)
    advance!(m, 60.0)
    push!(results, report_zeros("память сохраняет связь симметричной",
                                maximum(abs.(m.coupling .- m.coupling')), TOL_EXACT))

    println("\nмир внутри вибрации")

    seeds = 1:6   # всё, что ниже, сверяется при шести начальных условиях сразу

    # Мир — не копия вибрации: он смотрит на те же числа. Это не метафора, а то,
    # что можно сверить: после шага числа внутри и снаружи — одни и те же.
    m = build(modes = 3, band = 0.1, inner = 1.5, inner_memory = 1e-3)
    advance!(m, 5.0)
    gap = 0.0
    for i in 1:N
        w = world(m, i)
        gap = max(gap, maximum(abs.(vec(w.omega) .- m.omega[i, :])),
                       maximum(abs.(vec(w.theta) .- m.theta[i, :])))
    end
    push!(results, report_zeros("мир — не копия: его числа и числа снаружи одни", gap, TOL_EXACT))

    # Вибрация делает один шаг, а не по шагу на масштаб: силы мира и силы среды
    # складываются в один. Сумма частот при этом сохраняется — мир не двигает
    # среду целиком, как и среда не двигает себя целиком.
    worst = 0.0
    for seed in seeds
        m = build(modes = 3, band = 0.1, inner = 1.5, damping = 0.0, seed = seed)
        was = sum(m.omega)
        advance!(m, 20.0)
        worst = max(worst, abs(sum(m.omega) - was))
    end
    push!(results, report_zeros("силы масштабов складываются: сумма частот (γ = 0)", worst, TOL_EXACT))

    # Мир двигает числа только тем, что в нём есть связь: пустой мир — не мир.
    worst = 0.0
    for seed in seeds
        just = build(modes = 3, band = 0.1, memory = 0.0, seed = seed)
        just.inside = nothing
        empty_worlds = build(modes = 3, band = 0.1, inner = 1.5, memory = 0.0, seed = seed)
        for w in empty_worlds.inside
            w.coupling .= 0.0
        end
        advance!(just, 60.0)
        advance!(empty_worlds, 60.0)
        worst = max(worst, state_gap(just, empty_worlds))
    end
    push!(results, report_zeros("мир без связи ничего не делает: числа как без мира", worst, TOL_EXACT))

    # Строй — форма, а не среда: держать обертоны в отношении может только то, у
    # чего обертоны есть. У вибрации с одной модой держать нечего, и λ в закон не
    # входит вовсе.
    worst = 0.0
    for seed in seeds
        held = build(modes = 1, tuning = 8.0, seed = seed)
        loose = build(modes = 1, tuning = 0.0, seed = seed)
        advance!(held, 60.0)
        advance!(loose, 60.0)
        worst = max(worst, state_gap(held, loose))
    end
    push!(results, report_zeros("при одной моде строю держать нечего: λ не входит", worst, TOL_EXACT))

    # Строй живёт у вибрации, в её мире: у миров он свой, и среда снаружи его не
    # перебивает. Два описания, отличающиеся только строем среды, дают одни и те
    # же числа — и с мирами того же строя, и с мирами без строя. Что это не
    # пустая сверка, показывает контроль: у вибраций без мира тот же λ в закон
    # входит, и числа от него другие.
    inner_matrix = fill(1.5, 3, 3)
    for p in 1:3
        inner_matrix[p, p] = 0.0
    end
    worst, control = 0.0, Inf
    for seed in seeds
        a = build(modes = 3, band = 0.1, tuning = 2.0, memory = 0.0, seed = seed)
        inside!(a; coupling = inner_matrix, tuning = 2.0)
        b = build(modes = 3, band = 0.1, tuning = 1.0, memory = 0.0, seed = seed)
        inside!(b; coupling = inner_matrix, tuning = 2.0)
        c = build(modes = 3, band = 0.1, tuning = 2.0, memory = 0.0, seed = seed)
        inside!(c; coupling = inner_matrix, tuning = 0.0)
        d = build(modes = 3, band = 0.1, tuning = 0.0, memory = 0.0, seed = seed)
        inside!(d; coupling = inner_matrix, tuning = 0.0)
        for x in (a, b, c, d)
            advance!(x, 60.0)
        end
        worst = max(worst, state_gap(a, b), state_gap(c, d))

        bare = build(modes = 3, band = 0.1, tuning = 2.0, memory = 0.0, seed = seed)
        bare.inside = nothing
        other = build(modes = 3, band = 0.1, tuning = 1.0, memory = 0.0, seed = seed)
        other.inside = nothing
        advance!(bare, 60.0)
        advance!(other, 60.0)
        control = min(control, state_gap(bare, other))
    end
    push!(results, report_zeros("строй берётся у мира: снаружи λ в закон не входит", worst, TOL_EXACT))
    push!(results, report_at_least("а без мира λ входит: иначе и сверять нечего", control, 1e-3))

    # Русло внутри — то же, что русло между вибрациями: за шаг связь сдвигается к
    # совпадению фаз. Совпадение берётся в тот момент, когда оно пишется, — после
    # шага, — и закон сдвига сверяется с ним: Δ = dt·εᵤ·(совпадение − связь).
    worst = 0.0
    for seed in seeds
        m = build(modes = 3, band = 0.1, inner = 1.5, inner_memory = 1.0, memory = 0.0, seed = seed)
        was = inner_links(m)
        step!(m, DT)
        coin = [coincidence(m, i) for i in 1:N]
        now = inner_links(m)
        for i in 1:N
            worst = max(worst, abs(now[i] - was[i] - DT * 1.0 * (coin[i] - was[i])))
        end
    end
    push!(results, report_zeros("за шаг русло внутри идёт к совпадению: закон сдвига", worst, TOL_EXACT))

    # И пишет оно одни фазы: ни частота, ни строй в запись не входят. У одной и
    # той же пары мод с теми же фазами запись одна и та же — чем бы эти моды ни
    # звучали и каким бы строем ни держались. Контроль показывает, что различие,
    # которому не войти в запись, и вправду есть: частота сдвинута.
    worst, control = 0.0, Inf
    for seed in seeds
        a = build(modes = 3, band = 0.1, inner = 1.5, inner_memory = 1.0, memory = 0.0, seed = seed)
        b = build(modes = 3, band = 0.1, inner = 1.5, inner_memory = 1.0, memory = 0.0, seed = seed)
        c = build(modes = 3, band = 0.1, inner = 1.5, inner_memory = 1.0, memory = 0.0,
                  tuning = 8.0, seed = seed)
        perturb!(b, 1, 3.0)
        for i in 1:N
            inner_trace!(world(a, i), DT)
            inner_trace!(world(b, i), DT)
            inner_trace!(world(c, i), DT)
            worst = max(worst, maximum(abs.(a.inside[i].coupling .- b.inside[i].coupling)),
                               maximum(abs.(a.inside[i].coupling .- c.inside[i].coupling)))
        end
        control = min(control, maximum(abs.(a.omega .- b.omega)))
    end
    push!(results, report_zeros("след пишет одни фазы: ни частота, ни строй не входят", worst, TOL_EXACT))
    push!(results, report_at_least("а состояние разное: есть чему не войти в запись", control, 1e-3))

    println("\nграница среды")

    # Без нити узел вне среды в неё не входит: ничто в законе на него не смотрит,
    # и он остаётся при своей частоте. Мало того: и среда его не замечает — те же
    # двенадцать вибраций дают те же числа, что вовсе без гостя. Нулевые слагаемые
    # прибавляются точно, поэтому числа не «почти» одни, а одни.
    worst, frozen = 0.0, 0.0
    for seed in seeds
        m = chain(tie = 0.0, seed = seed)
        alone = Medium(omega0 = m.omega0[1:12, :], theta = m.theta[1:12, :],
                       coupling = m.coupling[1:12, 1:12], band = Inf, damping = 0.0)
        advance!(m, 20.0)
        advance!(alone, 20.0)
        worst = max(worst, maximum(abs.(m.omega[1:12, :] .- alone.omega)),
                           maximum(abs.(m.theta[1:12, :] .- alone.theta)))
        frozen = max(frozen, maximum(abs.(m.omega[13, :] .- m.omega0[13, :])))
    end
    push!(results, report_zeros("без нити узел вне среды не входит: частота стоит на своей", frozen, TOL_EXACT))
    push!(results, report_zeros("и среда гостя не замечает: её числа как без него", worst, TOL_EXACT))

    # Край — не граница: у среды с рваным краем связь всё ещё взаимна, и среду
    # по-прежнему нельзя сдвинуть целиком. А односторонняя нить — можно: она и
    # поставлена затем, чтобы было видно, что тяга есть и она не ноль.
    ragged = chain(reach = 3, tie = 0.0, band = Inf)
    push!(results, report_zeros("у среды с краем связь всё ещё взаимна: себя не сдвинуть",
                                abs(pull(ragged)), TOL_EXACT))
    one_sided = chain(reach = 3, tie = 3.2, gap = 0.5, band = 0.1, one_sided = true)
    push!(results, report_at_least("но не оттого, что тяги нет: односторонняя нить сдвигает",
                                   abs(pull(one_sided)), 1e-3))

    println("\nшов: две среды или одна с перемычкой")

    # Шов не имеет места: закон знает пару и её связь, а не место узла. Значит,
    # перемена места узла у шва и узла внутри ничего не меняет — за шаг это ровно
    # ноль. За минуту набирается 10⁻¹⁰: это уже не место, а порядок сложения, и
    # утверждение о среде тут ни при чём — поэтому сверяется шаг.
    perm = collect(1:12)
    perm[5], perm[7] = 7, 5
    worst, control = 0.0, Inf
    for seed in seeds
        m = lump_pair(tie = 0.6 / 6, gap = 0.1, seed = seed)
        p = Medium(omega0 = m.omega0[perm, :], theta = m.theta[perm, :],
                   coupling = m.coupling[perm, perm], band = Inf, damping = 0.0)
        step!(m, DT)
        step!(p, DT)
        worst = max(worst, maximum(abs.(p.omega .- m.omega[perm, :])),
                           maximum(abs.(p.theta .- m.theta[perm, :])))
        q = lump_pair(tie = 0.0, gap = 0.1, seed = seed)
        step!(q, DT)
        control = min(control, maximum(abs.(m.omega .- q.omega)))
    end
    push!(results, report_zeros("шов не имеет места: перемена места ничего не меняет", worst, TOL_EXACT))
    push!(results, report_at_least("а без перемычки те же две среды — другие числа", control, 1e-6))

    # Пока перемычка не проводит, среды две: два сгустка живут ровно так же, как
    # две среды порознь — теми же числами, а не похожими.
    worst = 0.0
    for seed in seeds
        m = lump_pair(tie = 0.0, gap = 0.1, seed = seed)
        a = Medium(omega0 = m.omega0[1:6, :], theta = m.theta[1:6, :],
                   coupling = full_link(0.6 / 6, 6), band = Inf, damping = 0.0)
        b = Medium(omega0 = m.omega0[7:12, :], theta = m.theta[7:12, :],
                   coupling = full_link(0.6 / 6, 6), band = Inf, damping = 0.0)
        advance!(m, 60.0)
        advance!(a, 60.0)
        advance!(b, 60.0)
        worst = max(worst, maximum(abs.(m.omega[1:6, :] .- a.omega)),
                           maximum(abs.(m.theta[1:6, :] .- a.theta)),
                           maximum(abs.(m.omega[7:12, :] .- b.omega)),
                           maximum(abs.(m.theta[7:12, :] .- b.theta)))
    end
    push!(results, report_zeros("перемычка 0 ⟹ два сгустка — две среды порознь", worst, TOL_EXACT))

    # Пересилить нельзя: перемычка сильнее нужной не сдвигает среду целиком —
    # она может перераспределить частоты, а общего тона силой не выдумывает.
    strong = lump_pair(tie = 12.8, gap = 0.2, band = 0.1)
    push!(results, report_zeros("пересилить нельзя: и при перемычке 12.8 себя не сдвинуть",
                                abs(pull(strong)), TOL_EXACT))

    println("\nстрой: чем держится аккорд")

    # Строй — не число среды, а натяжение, и равновесие обертона говорит, чему он
    # равен. λ держит только обертоны (p > 1), а основную частоту не держит вовсе —
    # связь тянет её наравне со всеми, и её уход входит в промах обертона
    # множителем p:
    #
    #     x_p = |D_p + γ·r₀_p − p·D₁| / ((λ + γ)·p·ω₁),
    #
    # где r₀_p — написанный промах, D_p — тяга связи на моду. Это то же число, что
    # читает сцена `tension.jl`, но здесь оно выведено из уравнений, а не найдено:
    # поэтому и проверяется сверкой, а не порогом.
    worst, short = 0.0, 0.0
    for lam in (8.0, 32.0), seed in 1:6
        m = build(modes = 3, band = 0.1, tuning = lam, seed = seed)
        advance!(m, 60.0)
        γ = m.damping
        for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
            base = abs(m.omega[i, 1])
            r0 = m.omega0[i, p] - p * m.omega0[i, 1]
            x = abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base)
            held = abs(drag(m, i, p) + γ * r0 - p * drag(m, i, 1)) / ((lam + γ) * p * base)
            cut = abs(drag(m, i, p) + γ * r0) / ((lam + γ) * p * base)
            worst = max(worst, abs(x - held))
            short = max(short, abs(x - cut))
        end
    end
    push!(results, report_zeros("промах обертона равен равновесию сил: |D_p + γ·r₀_p − p·D₁|/((λ+γ)·p·ω₁)",
                                worst, TOL_SCHEME))
    push!(results, report_at_least("а без тяги на основную частоту формула расходится", short, 1e-2))

    # Основная частота — не особенность нумерации, а то, что делает строй: закон
    # ничего не говорит о том, какая мода первая. При λ = 0 перестановка индексов
    # мод не меняет чисел — за шаг это ровно ноль. При λ > 0 меняет, и это не
    # изъян: перестановка и есть подмена того, что строй называет первым.
    perm = [2, 3, 1]
    back = invperm(perm)
    worst, control = 0.0, Inf
    for seed in 1:6
        a = build(modes = 3, band = 0.1, seed = seed)
        b = Medium(omega0 = a.omega0[:, perm], theta = a.theta[:, perm],
                   coupling = copy(a.coupling), band = a.band, damping = a.damping,
                   tuning = a.tuning, memory = a.memory)
        step!(a, DT)
        step!(b, DT)
        worst = max(worst, maximum(abs.(b.omega[:, back] .- a.omega)),
                           maximum(abs.(b.theta[:, back] .- a.theta)))
        c = build(modes = 3, band = 0.1, tuning = 8.0, seed = seed)
        d = Medium(omega0 = c.omega0[:, perm], theta = c.theta[:, perm],
                   coupling = copy(c.coupling), band = c.band, damping = c.damping,
                   tuning = c.tuning, memory = c.memory)
        step!(c, DT)
        step!(d, DT)
        control = min(control, maximum(abs.(d.omega[:, back] .- c.omega)),
                              maximum(abs.(d.theta[:, back] .- c.theta)))
    end
    push!(results, report_zeros("λ = 0: перестановка индексов мод не меняет чисел", worst, TOL_EXACT))
    push!(results, report_at_least("λ > 0: меняет — строй и делает первую моду первой", control, 1e-3))

    # Строй входит в закон не сам, а суммой с затуханием: у промаха обертона один
    # множитель на шаг — 1 − (λ+γ)·dt. Это видно прямо в записи: на ω_p действуют и
    # строй, и затухание, а на ω₁ — только затухание, поэтому у r = ω_p − p·ω₁ оба
    # слагаемых складываются. Проверка идёт по раскладам одной суммы (λ+γ = 16.8) и
    # прямо против шага: там, где γ велика, различие между суммой и одним λ·dt уже не
    # в последнем знаке — это и показывает второй контроль.
    worst, short = 0.0, 0.0
    for (lam, γ) in ((16.0, 0.8), (8.8, 8.0), (15.2, 1.6), (14.8, 2.0)), seed in 1:6
        m = build(modes = 3, band = 0.1, tuning = lam, damping = γ, seed = seed)
        advance!(m, 60.0)
        for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
            r0 = m.omega0[i, p] - p * m.omega0[i, 1]
            r = m.omega[i, p] - p * m.omega[i, 1]
            dr, d1 = drag(m, i, p), drag(m, i, 1)
            step!(m, DT)
            after = m.omega[i, p] - p * m.omega[i, 1]
            held = r + DT * (dr - p * d1 + γ * r0 - (lam + γ) * r)
            alone = r + DT * (dr - p * d1 + γ * r0 - lam * r)
            worst = max(worst, abs(after - held))
            short = max(short, abs(after - alone))
        end
    end
    push!(results, report_zeros("промах обертона множится за шаг на 1 − (λ+γ)·dt — у суммы один множитель",
                                worst, TOL_EXACT))
    push!(results, report_at_least("а у одного λ·dt (без γ в сумме) тот же множитель не сходится",
                                   short, 1e-4))

    # У основной частоты своя скорость, и строя в ней нет: строй держит обертоны
    # относительно неё, а её саму не держит. Значит, среда показывает два числа —
    # 1 − γ·dt у основной и 1 − (λ+γ)·dt у промаха, — и λ есть их разность, делённая
    # на шаг. Здесь проверяется первое: шага самой основной частоты строй не трогает,
    # а с суммой в её скорости тот же шаг не сходится — без этого разность читалась бы
    # по одному показанию, а не по двум.
    worst, short = 0.0, 0.0
    for (lam, γ) in ((16.0, 0.8), (8.8, 8.0), (15.2, 1.6)), seed in 1:6
        m = build(modes = 3, band = 0.1, tuning = lam, damping = γ, seed = seed)
        advance!(m, 60.0)
        for i in 1:size(m.omega, 1)
            w1 = m.omega[i, 1]
            d1 = drag(m, i, 1)
            step!(m, DT)
            after = m.omega[i, 1]
            alone = w1 + DT * (d1 - γ * (w1 - m.omega0[i, 1]))
            both = w1 + DT * (d1 - (lam + γ) * (w1 - m.omega0[i, 1]))
            worst = max(worst, abs(after - alone))
            short = max(short, abs(after - both))
        end
    end
    push!(results, report_zeros("у основной частоты своя скорость 1 − γ·dt: строй её не держит",
                                worst, TOL_EXACT))
    push!(results, report_at_least("а с суммой (λ+γ) в этой скорости тот же шаг не сходится",
                                   short, 1e-4))

    # И тогда λ читается по разности, а не по одному показанию: толчок в основную
    # частоту вибрации, и в одном опыте видно обе скорости — сама она возвращается
    # со своей, а промах, который толчок за собой тянет, спешит со своей. Читается
    # коротким окном: дальше в разность входит связь и её общая мода, и это уже не
    # строй (сцена `rates.jl`). Числа здесь измеренные, поэтому и допуск — не
    # машинный: проверяется, что разность даёт написанный λ, а при λ = 0 её нет.
    function rate_of(lam::Float64, γ::Float64, what; win::Int = 4, push::Float64 = 0.02,
                   seeds = 1:3)
        logs = Float64[]
        for s in seeds
            a = build(modes = 3, band = 0.1, tuning = lam, damping = γ, seed = s)
            b = build(modes = 3, band = 0.1, tuning = lam, damping = γ, seed = s)
            advance!(a, 5.0)
            advance!(b, 5.0)
            perturb!(b, 1, push, 1)
            before = what(b) - what(a)
            before == 0.0 && return NaN
            for _ in 1:win
                step!(a, DT)
                step!(b, DT)
            end
            after = what(b) - what(a)
            after != 0.0 || return NaN
            push!(logs, log(abs(after / before)) / win)
        end
        return exp(sum(logs) / length(logs))
    end
    own_of(m) = m.omega[1, 1]
    miss_of(m) = m.omega[1, 2] - 2 * m.omega[1, 1]
    read_lam, converge = 0.0, 0.0
    for (lam, γ) in ((16.0, 0.8), (8.8, 8.0), (15.2, 1.6))
        read = (rate_of(lam, γ, own_of) - rate_of(lam, γ, miss_of)) / DT
        read_lam = max(read_lam, abs(read - lam) / lam)
    end
    for γ in (0.8, 4.0, 8.0)
        converge = max(converge, abs(rate_of(0.0, γ, own_of) - rate_of(0.0, γ, miss_of)))
    end
    push!(results, report_zeros("λ читается разностью скоростей среды: (своя − промах)/dt",
                                read_lam, 3e-2))
    push!(results, report_zeros("при λ = 0 обе скорости — одно число: разность ноль",
                                converge, 1e-6))

    # Перевес (сцена `excess.jl`) — то, чем λ уносит разницу скоростей: тяга у мод
    # разная, и мера перевеса — доля формы, одна и та же в двух картинах. Утверждений
    # здесь три, и каждое — то, чего в сверке не было.
    #
    # Первое — предпосылка: тянут фазы, а не номера мод. Если бы тяга на обёртон была
    # ровно в p раз больше тяги на основную, разности D = D_p − p·D₁ не было бы вовсе,
    # и перевесу не с чем было бы спорить. Строка берётся там же, где её берёт сцена:
    # в месте наибольшего промаха (там, где доля и вправду доля).
    function miss_place(m::Medium)
        worst, at = 0.0, (1, 1)
        for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
            base = abs(m.omega[i, 1])
            base == 0.0 && continue
            x = abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base)
            x > worst && ((worst, at) = (x, (i, p)))
        end
        return at
    end
    least = Inf
    for (lam, γ) in ((16.0, 0.8), (8.8, 8.0), (15.2, 1.6), (2.0, 2.0)), seed in 1:6
        m = build(modes = 3, band = 0.1, tuning = lam, damping = γ, seed = seed)
        advance!(m, 60.0)
        i, p = miss_place(m)
        least = min(least, abs(drag(m, i, p) - p * drag(m, i, 1)))
    end
    push!(results, report_at_least("тянут фазы, а не номера мод: разность тяг D = D_p − p·D₁ не ноль",
                                    least, 1e-3))

    # Второе и третье — доля формы. Она читается двумя картинами: стоячей λ·r/D (силы
    # в стоячем состоянии) и скоростной λ/(λ+γ) (доли двух скоростей среды). Это одно
    # число только там, где написанного промаха нет: r₀ = 0 (аккорд написан точно).
    # Разность берётся тем же местом, что и выше; допуск здесь не машинный и не схемный:
    # невязка равновесия входит в долю делённой на D, а D на иных раскладах — десятая
    # доля, поэтому порог вчетверо шире схемного (TOL_SCHEME).
    function exact_chord(; seed::Int, tuning::Float64, damping::Float64)
        m = build(modes = 3, band = 0.1, tuning = tuning, damping = damping, seed = seed)
        for i in 1:size(m.omega0, 1), p in 2:size(m.omega0, 2)
            m.omega0[i, p] = p * m.omega0[i, 1]
        end
        m.omega .= m.omega0
        return m
    end
    writes = ((16.0, 0.8), (8.0, 1.0), (4.0, 4.0))
    shared, apart = 0.0, Inf
    for (lam, γ) in writes, seed in 1:6
        exact = exact_chord(seed = seed, tuning = lam, damping = γ)
        advance!(exact, 60.0)
        i, p = miss_place(exact)
        r = exact.omega[i, p] - p * exact.omega[i, 1]
        d = drag(exact, i, p) - p * drag(exact, i, 1)
        shared = max(shared, abs(lam * r / d - lam / (lam + γ)))

        written = build(modes = 3, band = 0.1, tuning = lam, damping = γ, seed = seed)
        advance!(written, 60.0)
        i, p = miss_place(written)
        r = written.omega[i, p] - p * written.omega[i, 1]
        d = drag(written, i, p) - p * drag(written, i, 1)
        apart = min(apart, abs(lam * r / d - lam / (lam + γ)))
    end
    push!(results, report_zeros("где написанного нет (r₀ = 0), стоячая доля λ·r/D равна скоростной λ/(λ+γ)",
                                shared, 2e-2))
    push!(results, report_at_least("а при написанном промахе те же две картины расходятся",
                                   apart, 1e-3))

    # И последнее место того же вопроса — само написанное: откуда берётся промах r₀, а
    # с ним и то, почему λ конечна (сцена `written.jl`). Здесь из этого четыре места.
    #
    # Первое: написанное не переписывается. Прожитое — события, память, ε у каждой
    # связи — меняет связи и не трогает своих частот ни на бит. Ноль здесь ровно ноль,
    # а не «почти»: этим и сказано, что r₀ не память и не след, а то, чем среда
    # написана. Второе утверждение — контроль: есть чему не переписаться, связи за
    # жизнь меняются.
    ε = fill(1e-3, N, N)
    ε[1, 2], ε[1, 3] = 1e-2, 1e-4
    rewrite, least_link = 0.0, Inf
    for seed in 1:6
        m = build(modes = 3, band = 0.1, memory_matrix = ε, seed = seed)
        was0, was_c = copy(m.omega0), copy(m.coupling)
        for _ in 1:200
            perturb!(m, 1, 1.5)
            advance!(m, 6.0)
        end
        rewrite = max(rewrite, maximum(abs.(m.omega0 .- was0)))
        least_link = min(least_link, maximum(abs.(m.coupling .- was_c)))
    end
    push!(results, report_zeros("написанное не переписывается: прожитое меняет связи, а свои частоты — нет",
                                rewrite, TOL_EXACT))
    push!(results, report_at_least("и не переписать его нечем: связи за прожитое меняются",
                                   least_link, 1e-2))

    # Третье и четвёртое — нужная λ и шаг. Аккорд написан заново, ровно 1 %: тогда
    # требование «промах не хуже написанного» — это доля 0.01, и λ_min = F/s − γ, где
    # F — тяга в долях отношения (то же число, что читает `tension.jl`). Проверяется
    # поведением с двух сторон: при половине нужной жёсткости написанное не держится,
    # при удвоенной — держится с запасом. Затем то же число читается шагом: промах не
    # сходит ниже F·dt/2 (потолок (λ+γ)·dt < 2 с другой стороны), и потому одно и то
    # же написанное 1 % при шаге 0.02 держится, а при 0.1 — нет. Счёт по зёрнам, а не
    # «наибольшее»: у одного зерна промах гуляет по времени и равновесия не ловит —
    # это уже читано (`rates.jl`, `excess.jl`), и превращать это в порог не стоит.
    function written_chord(; seed::Int, tuning::Float64)
        m = build(modes = 3, band = 0.1, link = 0.6, tuning = tuning, damping = 0.8,
                  memory = 1e-3, seed = seed)
        for i in 1:size(m.omega0, 1), p in 2:size(m.omega0, 2)
            m.omega0[i, p] = p * m.omega[i, 1] * (1 + WRITTEN)
        end
        m.omega .= m.omega0
        return m
    end
    function share_of_miss(m::Medium)
        worst = 0.0
        for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
            base = abs(m.omega[i, 1])
            base == 0.0 && continue
            worst = max(worst, abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base))
        end
        return worst
    end
    function force_over(m::Medium)
        worst = 0.0
        for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
            base = abs(m.omega[i, 1])
            base == 0.0 && continue
            r0 = m.omega0[i, p] - p * m.omega0[i, 1]
            worst = max(worst,
                        abs(drag(m, i, p) + m.damping * r0 - p * drag(m, i, 1)) / (p * base))
        end
        return worst
    end
    "Наименьшая жёсткость, при которой промах не хуже написанного: λ = F/s − γ."
    need_lambda(m::Medium) = force_over(m) / WRITTEN - m.damping
    function run_written(m::Medium, dt::Float64, seconds::Float64 = 60.0)
        for _ in 1:round(Int, seconds / dt)
            step!(m, dt)
        end
        return m
    end
    least_over, least_margin = Inf, Inf
    held_fine, held_coarse = 0, 0
    for seed in 1:6
        base = run_written(written_chord(seed = seed, tuning = 64.0), DT)
        lam = need_lambda(base)
        least_over = min(least_over,
                         share_of_miss(run_written(written_chord(seed = seed, tuning = lam / 2), DT)) / WRITTEN)
        least_margin = min(least_margin,
                           WRITTEN / share_of_miss(run_written(written_chord(seed = seed, tuning = 2 * lam), DT)))
        fine = run_written(written_chord(seed = seed, tuning = 2 / 0.02 - 0.8 - 0.5), 0.02)
        coarse = run_written(written_chord(seed = seed, tuning = 2 / 0.1 - 0.8 - 0.5), 0.1)
        held_fine += share_of_miss(fine) <= WRITTEN
        held_coarse += share_of_miss(coarse) <= WRITTEN
    end
    push!(results, report_at_least("при половине нужной жёсткости написанное не держится: промах выше во столько раз",
                                   least_over, 1.1))
    push!(results, report_at_least("а при удвоенной — держится: промах ниже написанного во столько раз",
                                   least_margin, 1.5))
    @printf("  %-56s %-11s %d из 6\n", "при шаге 0.02 написанное 1 % держится у зёрен",
            held_fine >= 4 ? "сошлось" : "НЕ СОШЛОСЬ", held_fine)
    push!(results, held_fine >= 4)
    @printf("  %-56s %-11s %d из 6\n", "а при шаге 0.1 — держится не больше чем у одного",
            held_coarse <= 1 ? "сошлось" : "НЕ СОШЛОСЬ", held_coarse)
    push!(results, held_coarse <= 1)


    println("\nглубина: что у моды внутри, кроме отношения")

    # Первое: мир моды пуст — и пуст ровным нулём. У моды одна вибрация, она сама;
    # среда есть взаимодействие вибраций, а одной вибрации взаимодействовать не с
    # чем. Второй уровень можно поставить той же записью, что и первый (`inside!`), и
    # он ничего не добавляет: ни силы, ни отличия. Значит глубина не растёт
    # повторением границы — на втором уровне повторять нечего.
    empty_pull, deep_gap = 0.0, 0.0
    for seed in 1:3
        deep = build(modes = 3, band = 0.1, tuning = 8.0, inner = 1.5, seed = seed)
        flat = build(modes = 3, band = 0.1, tuning = 8.0, inner = 1.5, seed = seed)
        for w in deep.inside
            inside!(w, coupling = fill(1.5, 1, 1))
        end
        force = zeros(1, 1)
        inner_force!(world(world(deep, 1), 1), force, 1)
        empty_pull = max(empty_pull, abs(force[1, 1]))
        advance!(deep, 60.0)
        advance!(flat, 60.0)
        deep_gap = max(deep_gap, state_gap(deep, flat),
                       maximum(abs.(deep.coupling .- flat.coupling)))
    end
    push!(results, report_zeros("мир моды пуст: у моды одна вибрация, тянуть не с чем",
                                empty_pull, TOL_EXACT))
    push!(results, report_zeros("и второй уровень не меняет ничего: две среды совпадают",
                                deep_gap, TOL_EXACT))

    # Второе: размер. Место решает, слышит ли мода соседа; размер — чем она держится.
    # Это не наблюдение, а следствие равновесия промаха:
    #
    #     x_p = |D_p + γ·r₀_p − p·D₁| / ((λ + γ)·p·ω₁).
    #
    # Если полоса едет вместе с аккордом (Π = Π·k при ω = ω·k), то S(Δ/Π) не
    # меняется ни на бит, и потому тяга D от размера не зависит вовсе; а растёт с ним
    # только написанное r₀_p = s·p·ω₀₁ и делитель p·ω₁. Дальше — по строкам: у моды
    # (i, p) своя сила F = |c + γ·r₀_p|/(p·ω₁), где c = D_p − p·D₁ — тот самый перевес,
    # который читает `excess.jl`. Там, где среда тянет обертон от основной (c > 0),
    # модуль раскрывается и γ из нужной жёсткости уходит целиком:
    #
    #     λ_нуж = F/s − γ = c / (s·p·ω₁)   — тяга перевеса, делённая на написанную
    #                                         долю и на размер моды.
    #
    # Отсюда и закон: λ_нуж(k)·k = λ_нуж(1) — точный, а не приблизительный. Сверяется
    # на одном состоянии: две среды одного зерна, у одной размер 1, у другой k, —
    # различий динамики здесь ещё нет и быть не должно. И то же с другой стороны: у
    # наибольшего промаха строка-носитель со размером может смениться — тогда λ·k
    # уплывает, и это печатается числом (сколько зёрен, во сколько раз), а не порогом.
    "Нужная жёсткость моды (i, p) там, где среда тянет обертон от основной."
    function excess_need(m::Medium, i::Int, p::Int)
        c = drag(m, i, p) - p * drag(m, i, 1)
        c <= 0.0 && return NaN
        return c / (WRITTEN * p * abs(m.omega[i, 1]))
    end
    "То же по строке (i, p) из записи: F/s − γ."
    function row_need(m::Medium, i::Int, p::Int)
        base = abs(m.omega[i, 1])
        r0 = m.omega0[i, p] - p * m.omega0[i, 1]
        f = abs(drag(m, i, p) + m.damping * r0 - p * drag(m, i, 1)) / (p * base)
        return f / WRITTEN - m.damping
    end
    "Строка (i, p), где сила наибольшая: её и читает нужная жёсткость."
    function carrier_row(m::Medium)
        worst, row = -Inf, (0, 0)
        for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
            base = abs(m.omega[i, 1])
            base == 0.0 && continue
            r0 = m.omega0[i, p] - p * m.omega0[i, 1]
            f = abs(drag(m, i, p) + m.damping * r0 - p * drag(m, i, 1)) / (p * base)
            f > worst && ((worst, row) = (f, (i, p)))
        end
        return row
    end
    tug_gap, sum_gap, law_gap, checked = 0.0, 0.0, 0.0, 0
    scattered, worst_spread = 0, 0.0
    for seed in 1:6
        needs = Float64[]
        first_row, turned = (0, 0), false
        for k in (0.25, 0.5, 1.0, 2.0, 4.0)
            one = written_chord(seed = seed, tuning = 8.0)
            many = written_chord(seed = seed, tuning = 8.0)
            many.band *= k
            many.omega0 .*= k
            many.omega .*= k
            for i in 1:size(one.omega, 1), p in 1:size(one.omega, 2)
                tug_gap = max(tug_gap, abs(drag(many, i, p) - drag(one, i, p)))
            end
            for i in 1:size(one.omega, 1), p in 2:size(one.omega, 2)
                need = excess_need(one, i, p)
                isnan(need) && continue
                checked += 1
                sum_gap = max(sum_gap, abs(row_need(one, i, p) - need))
                law_gap = max(law_gap, abs(excess_need(many, i, p) - need / k))
            end
            first_row == (0, 0) && (first_row = carrier_row(many))
            turned = turned || carrier_row(many) != first_row
            push!(needs, need_lambda(many) * k)
        end
        spread = maximum(needs) / minimum(needs)
        if turned
            scattered += 1
            worst_spread = max(worst_spread, spread)
        end
    end
    push!(results, report_zeros("тяга связи от размера не зависит, когда полоса едет с ним",
                                tug_gap, TOL_EXACT))
    push!(results, report_zeros("нужная жёсткость есть тяга перевеса, делённая на написанное и на размер",
                                sum_gap, TOL_EXACT))
    push!(results, report_zeros("а размеру она обратна точно: λ_нуж(k)·k = λ_нуж(1)",
                                law_gap, TOL_EXACT))
    @printf("  %-56s %-11s %d\n", "сверено мест, где среда тянет обертон от основной",
            "число", checked)
    if scattered == 0
        @printf("  %-56s %-11s у 0 из 6, носитель один\n",
                "а по наибольшему промаху λ·k постоянно", "число")
    else
        @printf("  %-56s %-11s у %d из 6, до %.2f раза\n",
                "а по наибольшему промаху носитель меняется, и λ·k уплывает", "число",
                scattered, worst_spread)
    end

    # Третье — то же самое, но поведением, и с другого конца: там сверялись два
    # состояния одного зерна, а здесь два размера проживают минуту сами. Доли промаха
    # у них расходятся, и это не спор с первым: размер у среды не прячется. Частоты и
    # тяга растут вместе с размером, а время — нет: у крупного аккорда фазы уходят
    # вперёд во столько же раз, и совпадения в среде другие. Порог объявлен десятой
    # долей процента — ниже самого расхождения на порядок, — и берётся по зёрнам:
    # у одного зерна промах гуляет по времени, и порогом это не лечится.
    gap_size = Inf
    for seed in 1:6
        shares = Float64[]
        for k in (0.25, 1.0, 4.0)
            m = written_chord(seed = seed, tuning = 8.0)
            m.band *= k
            m.omega0 .*= k
            m.omega .*= k
            run_written(m, DT)
            push!(shares, share_of_miss(m))
        end
        gap_size = min(gap_size, maximum(shares) - minimum(shares))
    end
    push!(results, report_at_least("а в поведении размер виден: доли промаха у размеров расходятся",
                                   gap_size, 1e-2))

    # Четвёртое — обратный перевес: то, чего здесь до сих пор не читалось ни разу.
    # Первое чтение отбрасывает строки, которые среда тянет обертон К основной
    # (`excess_need` возвращает NaN): 315 мест сверено, 285 нет. Спрашивается то же
    # число без отбрасывания. Модуль раскрывается по знаку перевеса:
    #
    #     c > 0:  λ_нуж = c/(s·p·ω₁)                    — γ уходит целиком (сверено);
    #     c < 0:  λ_нуж = (|c| − 2γ·r₀_p)/(s·p·ω₁)      при |c| > γ·r₀_p;
    #             λ_нуж = −|c|/(s·p·ω₁)                 при |c| < γ·r₀_p — знак минус.
    #
    # γ входит в нужную жёсткость дважды: возвратом к написанному в числителе и
    # вычетом в самом определении. При c > 0 оба входа гасят друг друга; при c < 0
    # складываются, и тогда «обратна размеру» перестаёт быть законом моды: на таких
    # строках λ_нуж·k идёт по прямой и падает ровно на 2γ за единицу размера. И
    # неположительная нужная жёсткость — это не послабление, а признание, что
    # требование на строке уже выполнено: она значит ровно, что промах равновесия
    # при λ = 0 не хуже написанного. Граница этого — не «перевес по модулю», а
    # перевес от нуля до минус удвоенного возврата, −2γ·r₀_p ≤ c ≤ 0: строка
    # держится сама, только если среда тянет её к основной и не сильнее, чем
    # возврат затухания её же написанного. Границы эти ходят с размером (r₀_p ∝ k),
    # а перевес — нет.
    "Сила строки (i, p) в долях отношения: |c + γ·r₀_p|/(p·ω₁) — запись равновесия."
    function row_force(m::Medium, i::Int, p::Int)
        base = abs(m.omega[i, 1])
        r0 = m.omega0[i, p] - p * m.omega0[i, 1]
        return abs(drag(m, i, p) + m.damping * r0 - p * drag(m, i, 1)) / (p * base)
    end
    branch_gap, dive_gap = 0.0, 0.0
    all_rows, toward_rows, free_rows, free_agree = 0, 0, 0, 0
    rows_by_need, rows_by_border, eligible_rows = 0, 0, 0
    neg_carriers, all_carriers = 0, 0
    for seed in 1:6
        first_needs, eligible = Float64[], Bool[]
        for k in (1.0, 0.25, 0.5, 2.0, 4.0)
            m = written_chord(seed = seed, tuning = 8.0)
            m.band *= k
            m.omega0 .*= k
            m.omega .*= k
            idx = 0
            for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
                idx += 1
                base = abs(m.omega[i, 1])
                c = drag(m, i, p) - p * drag(m, i, 1)
                r0 = m.omega0[i, p] - p * m.omega0[i, 1]
                need = row_need(m, i, p)
                all_rows += 1
                c <= 0.0 && (toward_rows += 1)
                if c > 0.0
                    branch_gap = max(branch_gap, abs(need - c / (WRITTEN * p * base)))
                elseif abs(c) > m.damping * r0
                    branch_gap = max(branch_gap,
                                     abs(need - (abs(c) - 2 * m.damping * r0) / (WRITTEN * p * base)))
                else
                    branch_gap = max(branch_gap, abs(need + abs(c) / (WRITTEN * p * base)))
                end
                need <= 0.0 && (rows_by_need += 1)
                (-2 * m.damping * r0 <= c <= 0.0) && (rows_by_border += 1)
                need <= 0.0 && (free_rows += 1)
                free_agree += (need <= 0.0) == (row_force(m, i, p) / m.damping <= WRITTEN)
                if k == 1.0
                    push!(first_needs, need)
                    # Одну ветвь на всех размерах даёт только строка «к», у которой перевес
                    # больше возврата и при самом крупном размере: возврат растёт с ним.
                    push!(eligible, c < 0.0 && abs(c) > 4 * m.damping * r0)
                elseif idx <= length(eligible) && eligible[idx]
                    eligible_rows += 1
                    dive_gap = max(dive_gap,
                                   abs(need * k - first_needs[idx] + 2 * m.damping * (k - 1)))
                end
            end
        end
        let m0 = written_chord(seed = seed, tuning = 8.0)
            (i, p) = carrier_row(m0)
            all_carriers += 1
            drag(m0, i, p) - p * drag(m0, i, 1) <= 0.0 && (neg_carriers += 1)
        end
    end
    push!(results, report_zeros("нужная жёсткость у моды одна на оба знака перевеса",
                                branch_gap, TOL_EXACT))
    push!(results, report_zeros("а где среда складывает перевесом больше возврата, λ_нуж·k падает на 2γ",
                                dive_gap, TOL_EXACT))
    push!(results, report_zeros("и граница у этого — от нуля до минус удвоенного возврата",
                                Float64(abs(rows_by_need - rows_by_border)), TOL_EXACT))
    @printf("  %-56s %-11s %d из %d\n",
            "неположительная нужная жёсткость значит требование уже выполнено",
            free_agree == all_rows ? "сошлось" : "НЕ СОШЛОСЬ", free_rows, all_rows)
    push!(results, free_agree == all_rows)
    @printf("  %-56s %-11s %d из %d\n", "из них строк «к» — тех, что сверка отбрасывала",
            "число", toward_rows, all_rows)
    @printf("  %-56s %-11s %d из %d\n", "и носителей в этой половине",
            "число", neg_carriers, all_carriers)
    @printf("  %-56s %-11s %d\n", "сверено строк, у которых ветвь одна на всех размерах",
            "число", eligible_rows)

    return results
end

"""
    mean_link(medium)

Средняя связь по всем связям, кроме собственных: то, что связь накопила. У свежей
среды это её начальные 0.6/N = 0.060 — исходная связь, а не накопленное.
"""
function mean_link(m::Medium)
    s, n = 0.0, 0
    for i in axes(m.coupling, 1), j in axes(m.coupling, 2)
        i == j && continue
        s += m.coupling[i, j]
        n += 1
    end
    return s / n
end

"""
    relief_of(a, b)

Рельеф: чем связь среды `a` отличается от связи среды `b` по каждому месту.
Собственные связи не считаются: там своя связь, и она не пишется.
"""
function relief_of(a::Medium, b::Medium)
    δ = zeros(size(a.coupling))
    for i in axes(δ, 1), j in axes(δ, 2)
        i == j && continue
        δ[i, j] = a.coupling[i, j] - b.coupling[i, j]
    end
    return δ
end

"""
    relief_place(δ, k)

Рельеф у места k (связи, которых k касается) и вдали от него: среднее |δ| по
связям. Место — положение, а не величина: если запись помнит место, у места
должно быть больше.
"""
function relief_place(δ::AbstractMatrix{Float64}, k::Int)
    near, far, nn, nf = 0.0, 0.0, 0, 0
    for i in axes(δ, 1), j in axes(δ, 2)
        i == j && continue
        v = abs(δ[i, j])
        if i == k || j == k
            near += v; nn += 1
        else
            far += v; nf += 1
        end
    end
    return (near / nn, far / nf)
end

"""
    relief_agreement(δa, δb)

Похожесть двух рельефов: связь Пирсона по местам. Единица — рельеф тот же,
ноль — связи между рельефами нет.
"""
function relief_agreement(δa::AbstractMatrix{Float64}, δb::AbstractMatrix{Float64})
    va, vb = Float64[], Float64[]
    for i in axes(δa, 1), j in axes(δa, 2)
        i == j && continue
        push!(va, δa[i, j])
        push!(vb, δb[i, j])
    end
    return cor(va, vb)
end

"""
    measured_over(seed; passes = 200)

Одно и то же событие в трёх средах при одном начальном условии: свежая, прожившая
`passes` проходов без событий и прожившая те же проходы с событием. Третья нужна
затем, что от одного лишь прожитого времени среда меняется: без неё различие
свежей и прожившей выдавалось бы за действие события.

Отдельное число здесь ничего не значит: у другого начального условия оно
другое, и разница между версиями Julia видна именно на этих числах.
"""
function measured_over(seed::Int; passes::Int = 200)
    fresh = build(memory = 1e-3, seed = seed)
    quiet = build(memory = 1e-3, seed = seed)
    routed = build(memory = 1e-3, seed = seed)
    for _ in 1:passes
        perturb!(routed, 1, 1.5)
        advance!(routed, 6.0)
        advance!(quiet, 6.0)
    end
    a, q, b = response(fresh), response(quiet), response(routed)
    return (link = (mean_link(fresh), mean_link(quiet), mean_link(routed)),
            half = (a.half_time, q.half_time, b.half_time),
            shape = (a.shape, q.shape, b.shape),
            residual = (a.residual, q.residual, b.residual),
            echo = (echo(build(memory = 1e-3, seed = seed)), echo(quiet), echo(routed)))
end

"""
    spread_of(pairs)

Размах по начальным условиям: наименьшее и наибольшее из измеренного.
"""
spread_of(pairs) = (minimum(first.(pairs)), maximum(last.(pairs)))

"""
    behaviour()

Что меняется в поведении — числа, не утверждения. И не одно число, а размах
по шести начальным условиям: иначе бросок генератора выдаётся за свойство
среды. Устойчиво здесь то, что не перекрывается при смене начальных условий.

Среды три, и различаются они по-разному: свежая и прожившая без событий —
прожитым временем, прожившая без событий и с руслом — событием. Без третьего
столбца событию приписывалось бы то, что делает прожитое.
"""
function behaviour()
    println("\nчто меняется в поведении (числа, не утверждения)")
    println("  шесть начальных условий; в скобках — наименьшее … наибольшее")
    println("  свежая — не прожившая ничего; без событий — прожившая столько же без")
    println("  событий; с руслом — прожившая те же проходы с событием")

    rows = [measured_over(seed) for seed in 1:6]

    for (name, field, unit) in (("связь средняя", :link, ""),
                                ("половина пути", :half, " с"),
                                ("отличие от затухания", :shape, ""),
                                ("остаток после возврата", :residual, ""),
                                ("отзвук в 5-й вибрации", :echo, ""))
        cols = [spread_of([getfield(r, field)[k] for r in rows]) for k in 1:3]
        @printf("  %-22s свежая %.4f … %.4f%s\n", name, cols[1]..., unit)
        @printf("  %-22s без событий %.4f … %.4f%s\n", "", cols[2]..., unit)
        @printf("  %-22s с руслом %.4f … %.4f%s\n", "", cols[3]..., unit)
        spent, quiet_w, rout_w = ((c[2] - c[1]) for c in cols)
        @printf("  %-22s разброс %.4f / %.4f / %.4f (свежая / без событий / с руслом)\n",
                "", spent, quiet_w, rout_w)
        # Диапазоны прожившей без событий и с руслом: перекрытие значит, что
        # событие в этой мере ничего не добавляет к прожитому.
        same = !(cols[2][2] < cols[3][1] || cols[3][2] < cols[2][1])
        println("  "^22 * (same ? "прожившая без событий и с руслом: диапазоны перекрываются"
                            : "прожившая без событий и с руслом: диапазоны разошлись"))
        # Сужение разброса — то, что держится и по начальным условиям, и по
        # версии. Печатается только там, где сравниваются сопоставимые меры:
        # свежая и прожившая без событий отличаются прожитым, а не событием.
        if quiet_w == 0.0
            println("  "^22 * "сужения нет: прожившая без событий не гуляет (разброс 0.0000)")
        elseif spent == 0.0
            println("  "^22 * "сужения нет: свежая мера не гуляет (разброс 0.0000)")
        elseif spent >= 2 * quiet_w
            @printf("  %-22s сужение есть: разброс уже в %.0f раз — его даёт прожитое\n",
                    "", spent / quiet_w)
        else
            @printf("  %-22s сужения нет: разброс прожившей %.2f нетронутого\n", "",
                    quiet_w / spent)
        end
    end
    println("  Перекрытие диапазонов значит, что различие сред не держится при")
    println("  смене начальных условий: такое число — бросок, а не свойство.")
    println("  Где прожившая без событий и прожившая с руслом сходятся, различие")
    println("  относится к прожитому, а не к событию: событие входит в запись")
    println("  только через состояние, а память пишет прожитое.")
    println("  Сужение разброса держится и по начальным условиям, и по версии там,")
    println("  где мера и есть отклик, отобранный памятью, — в отзвуке.")
    return nothing
end

function main()
    println("Сходится ли описание\n")
    results = checks()
    behaviour()
    println()
    if all(results)
        @printf("описание сходится: %d утверждений из %d\n", count(results), length(results))
        return 0
    end
    @printf("НЕ СОШЛОСЬ: %d из %d\n", length(results) - count(results), length(results))
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

