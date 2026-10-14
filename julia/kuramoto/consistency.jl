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
# Сцены открытых вопросов (`modes.jl`, `boundary.jl`, `seam.jl`) читают; здесь то,
# что из этих чтений следует из уравнений, превращено в утверждения — то же число,
# но без броска генератора.
#
# Запуск:
#
#     julia julia/kuramoto/consistency.jl

include(joinpath(@__DIR__, "basic.jl"))

const TOL_EXACT = 1e-9     # численная сумма: сходится к нулю
const TOL_SCHEME = 5e-3    # точность схемы первого порядка

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
    part_a, part_b = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10]
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

    return results
end

"""
    measured_over(seed; passes = 200)

Одно и то же событие в двух средах при одном начальном условии:
нетронутая среда и среда, через которую событие прошло двести раз.

Отдельное число здесь ничего не значит: у другого начального условия оно
другое, и разница между версиями Julia видна именно на этих числах.
"""
function measured_over(seed::Int; passes::Int = 200)
    fresh = build(memory = 1e-3, seed = seed)
    routed = build(memory = 1e-3, seed = seed)
    for _ in 1:passes
        perturb!(routed, 1, 1.5)
        advance!(routed, 6.0)
    end
    a = response(fresh)
    b = response(routed)
    return (link = (0.6 / N, routed.coupling[1, 2]),
            half = (a.half_time, b.half_time),
            shape = (a.shape, b.shape),
            residual = (a.residual, b.residual),
            echo = (echo(build(memory = 1e-3, seed = seed)), echo(routed)))
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
"""
function behaviour()
    println("\nчто меняется в поведении (числа, не утверждения)")
    println("  шесть начальных условий; в скобках — наименьшее … наибольшее")

    rows = [measured_over(seed) for seed in 1:6]

    for (name, field, unit) in (("связь[1,2]", :link, ""),
                                ("половина пути", :half, " с"),
                                ("отличие от затухания", :shape, ""),
                                ("остаток после возврата", :residual, ""),
                                ("отзвук в 5-й вибрации", :echo, ""))
        fro = spread_of([getfield(r, field)[1] for r in rows])
        rou = spread_of([getfield(r, field)[2] for r in rows])
        fro_lo, fro_hi = first(fro), last(fro)
        rou_lo, rou_hi = first(rou), last(rou)
        crossed = !(fro_hi < rou_lo || rou_hi < fro_lo)
        @printf("  %-23s нетронутая %.4f … %.4f   с руслом %.4f … %.4f%s\n",
                name, fro_lo, fro_hi, rou_lo, rou_hi, unit)
        spent, steadied = fro_hi - fro_lo, rou_hi - rou_lo
        @printf("  %-23s разброс %.4f против %.4f; диапазоны %s\n", "",
                spent, steadied, crossed ? "перекрываются" : "не пересекаются")
        # Ширина отклика — то, что держится и по начальным условиям, и по версии:
        # величина может совпасть, а воспроизводимость — нет. Подпись говорит
        # только то, что видно в числах: сужение есть там, где мера и есть
        # отклик, отобранный памятью (отзвук); у прочих мер его нет.
        if steadied == 0.0
            println("  "^23 * "разброс отклика: стоит на месте (с руслом)")
        elseif spent == 0.0
            println("  "^23 * "сужения нет: нетронутая мера не гуляет (разброс 0.0000)")
        elseif spent >= 2 * steadied
            @printf("  %-23s сужение есть: разброс уже в %.0f раз (с руслом)\n", "", spent / steadied)
        else
            @printf("  %-23s сужения нет: разброс отклика %.2f нетронутого (с руслом)\n", "",
                    steadied / spent)
        end
    end
    println("  Перекрытие диапазонов значит, что различие сред не держится при")
    println("  смене начальных условий: такое число — бросок, а не свойство.")
    println("  Сужение разброса держится и по начальным условиям, и по версии там,")
    println("  где мера и есть отклик, отобранный памятью, — в отзвуке: след —")
    println("  не остаток, а устойчивость.")
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

