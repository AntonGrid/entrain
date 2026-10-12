# Entrain. Курамото: десять осцилляторов.
#
# Это описание, а не проверка.
# Здесь нет порогов, гипотез, вердиктов и «а что если».
# Здесь есть:
#
#     состояние  — частота      (не число, не запись)
#     связь      — coupling     (как одна вибрация тянет другую)
#     затухание  — возвращение к своей частоте
#     резонанс   — совпадение частот
#     интерференция — наложение волн
#     память     — деформация связи (русло)
#
# Запуск:
#
#     julia julia/kuramoto/basic.jl

using Printf
using Random
using Statistics

const N = 10        # сколько вибраций в среде
const DT = 0.01     # шаг описания, с
const STEPS_6S = 600

"""
    Medium

Среда — не набор объектов, а взаимодействие вибраций.

- `omega0`   — своя частота каждой вибрации (чем она была бы одна);
- `omega`    — частота сейчас: это и есть состояние;
- `theta`    — фаза;
- `coupling` — связь: `coupling[i, j]` — насколько j тянет i;
- `memory`   — как быстро связь деформируется тем, что через неё прошло;
- `damping`  — как быстро частота возвращается к своей (затухание).
"""
mutable struct Medium
    omega0::Vector{Float64}
    omega::Vector{Float64}
    theta::Vector{Float64}
    coupling::Matrix{Float64}
    memory::Float64
    damping::Float64
end

"""
    step!(medium, dt)

Один шаг: связь двигает частоты, затухание возвращает их к своим,
прошедшее оставляет след в связи.

    θ̇ᵢ = ωᵢ + Σⱼ Wᵢⱼ sin(θⱼ − θᵢ)
    ω̇ᵢ = Σⱼ Wᵢⱼ sin(θⱼ − θᵢ) − γ (ωᵢ − ω₀ᵢ)
    Ẇᵢⱼ = ε (cos(θᵢ − θⱼ) − Wᵢⱼ)
"""
function step!(m::Medium, dt::Float64)
    n = length(m.theta)
    drive = zeros(n)
    for i in 1:n, j in 1:n
        drive[i] += m.coupling[i, j] * sin(m.theta[j] - m.theta[i])
    end
    # состояние — частота: связь её двигает, затухание возвращает к своей
    m.omega .+= dt .* (drive .- m.damping .* (m.omega .- m.omega0))
    m.theta .+= dt .* m.omega
    # память: связь деформируется тем, что прошло через неё
    m.coupling .+= dt .* m.memory .* (cos.(m.theta .- m.theta') .- m.coupling)
    return m
end

"""
    coherence(medium)

Согласованность: насколько вибрации звучат вместе. Ноль — каждая сама по себе,
единица — как одна. Это не оценка и не балл, а описание состояния среды.
"""
function coherence(m::Medium)
    z = mean(exp.(im .* m.theta))
    return abs(z), angle(z)
end

"""
    resonance(medium)

Резонанс — совпадение частот. Разброс частот равен нулю, когда все вибрации
звучат на одной частоте.
"""
resonance(m::Medium) = maximum(m.omega) - minimum(m.omega)

"""
    wave(medium, t)

Интерференция — наложение: то, что слышно в точке среды в момент `t`.
Сумма всех колебаний, а не какое-то из них.
"""
wave(m::Medium, t::Float64) = mean(cos.(m.omega .* t .+ m.theta))

"""
    perturb!(medium, index, shift)

Возмущение: у одной вибрации сбита частота. Событие.
"""
function perturb!(m::Medium, index::Int, shift::Float64)
    m.omega[index] += shift
    return m
end

"""
    drift(medium, index)

Насколько частота отклонилась от своей. Так видно затухание:
частота возвращается к ω₀, потому что среда не помнит ничего другого.
"""
drift(m::Medium, index::Int) = abs(m.omega[index] - m.omega0[index])

"""
    build(; memory, damping, seed)

Собрать среду из десяти вибраций. Частоты у них разные: среда не настроена.
Дальше всё, что происходит, — следствие этих чисел, а не чья-то воля.
"""
function build(; memory::Float64 = 1e-3, damping::Float64 = 0.8, seed::Int = 4)
    rng = MersenneTwister(seed)
    omega0 = 1.0 .+ 0.15 .* randn(rng, N)
    theta = 2π .* rand(rng, N)
    coupling = fill(0.6 / N, N, N)
    return Medium(omega0, copy(omega0), theta, coupling, memory, damping)
end

"""
    advance!(medium, seconds)

Дать среде звучать указанное время.
"""
function advance!(m::Medium, seconds::Float64)
    for _ in 1:round(Int, seconds / DT)
        step!(m, DT)
    end
    return m
end

"""
    show(medium, label)

Описать состояние среды словами и числами.
"""
function show(m::Medium, label::AbstractString)
    ρ, _ = coherence(m)
    @printf(
        "%-30s согласованность %.3f   разброс частот %.3f   связь (средняя) %.4f\n",
        label, ρ, resonance(m), mean(m.coupling),
    )
    return m
end

"""
    main()

Описание в четырёх сценах: как среда приходит к согласованности; что слышно
в точке (наложение); как толчок уходит (затухание); как среда помнит (русло).
"""
function main()
    @printf("десять вибраций, %d шагов в секунду описания\n\n", round(Int, 1 / DT))

    m = build()
    show(m, "сразу")
    advance!(m, 60.0)
    show(m, "через минуту")

    println("\nналожение: что слышно в одной точке среды")
    for t in (0.0, 0.5, 1.0, 2.0)
        @printf("   момент %4.1f с   %.4f\n", t, wave(m, t))
    end

    println("\nзатухание: толчок и возвращение к своей частоте")
    perturb!(m, 1, 1.5)
    @printf("   после толчка        отклонение %.3f\n", drift(m, 1))
    advance!(m, 6.0)
    @printf("   через шесть секунд  отклонение %.3f\n", drift(m, 1))

    println("\nпамять: связь деформируется тем, что через неё прошло")
    untouched = build()
    routed = build()
    @printf("   связь[1,2] у нетронутой среды     %.6f\n", untouched.coupling[1, 2])
    for _ in 1:200
        perturb!(routed, 1, 1.5)
        advance!(routed, 6.0)
    end
    @printf("   связь[1,2] у слышавшей двести раз %.6f\n", routed.coupling[1, 2])
    show(untouched, "   нетронутая среда")
    show(routed, "   среда с руслом")

    println("\nодно и то же событие в двух средах")
    perturb!(untouched, 1, 1.5)
    perturb!(routed, 1, 1.5)
    advance!(untouched, 6.0)
    advance!(routed, 6.0)
    @printf("   нетронутая: отклонение %.3f\n", drift(untouched, 1))
    @printf("   с руслом:   отклонение %.3f\n", drift(routed, 1))
    println("\nрусло — это не запись о событии. Это среда, которая стала другой.")

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
