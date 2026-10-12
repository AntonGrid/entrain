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
        # величина может совпасть, а воспроизводимость — нет.
        steadied == 0.0 && println("  "^23 * "разброс отклика: стоит на месте (с руслом)")
        steadied > 0.0 && @printf("  %-23s разброс отклика уже в %.0f раз (с руслом)\n", "",
                                  spent / steadied)
    end
    println("  Перекрытие диапазонов значит, что различие сред не держится при")
    println("  смене начальных условий: такое число — бросок, а не свойство.")
    println("  А ширина отклика держится: след — не остаток, а устойчивость.")
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

