# Entrain. Сходится ли описание.
#
# Это не опыт и не проверка мира. Утверждения описания сверяются друг с другом:
# то, что следует из уравнений, сравнивается с тем, что из них считается.
# Порогов здесь нет: есть точность чисел с плавающей точкой и точность схемы.
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
listen_to(m::Medium, indices, t::Float64) = mean(cos.(m.omega[indices] .* t .+ m.theta[indices]))

"""
    checks()

Все утверждения описания, которые можно сверить числами.
"""
function checks()
    results = Bool[]

    println("связь и состояние")

    m = build()
    total = 0.0
    for i in eachindex(m.theta), j in eachindex(m.theta)
        total += m.coupling[i, j] * sin(m.theta[j] - m.theta[i])
    end
    push!(results, report_zeros("Σᵢⱼ Wᵢⱼ sin(θⱼ−θᵢ) = 0 при симметричной связи", abs(total), TOL_EXACT))

    m = build(damping = 0.0)
    before = sum(m.omega)
    advance!(m, 20.0)
    push!(results, report_zeros("сумма частот сохраняется (γ = 0)", abs(sum(m.omega) - before), TOL_EXACT))

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
    m.theta .= m.theta[1]
    push!(results, report_zeros("все фазы равны ⟹ согласованность = 1", abs(coherence(m)[1] - 1.0), TOL_EXACT))

    m.theta .= 2π .* (0:(N - 1)) ./ N
    push!(results, report_zeros("фазы разложены равномерно ⟹ согласованность = 0", coherence(m)[1], TOL_EXACT))

    m = build()
    ρ = coherence(m)[1]
    m.theta .+= 1.2345
    push!(results, report_zeros("общий сдвиг фаз не меняет согласованность", abs(coherence(m)[1] - ρ), TOL_EXACT))

    m = build()
    part_a, part_b = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10]
    whole = wave(m, 1.7)
    split = (length(part_a) * listen_to(m, part_a, 1.7) + length(part_b) * listen_to(m, part_b, 1.7)) / N
    push!(results, report_zeros("наложение: целое = взвешенная сумма частей", abs(whole - split), TOL_EXACT))

    return results
end

function main()
    println("Сходится ли описание\n")
    results = checks()
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

