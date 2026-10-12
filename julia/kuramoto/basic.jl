# Entrain. Курамото: десять вибраций.
#
# Это описание, а не проверка.
# Здесь нет порогов, гипотез, вердиктов и «а что если».
#
# Записи:
#
#     состояние — частота; у одной вибрации их несколько: аккорд
#     связь     — coupling: как одна вибрация тянет другую
#     резонанс  — отклик по совпадению частот: полоса Π широкая — связь
#                 отзывается на многое, узкая — только на совпадающее
#     строй     — жёсткость аккорда: держит обертоны в их отношении
#     затухание — возвращение к своим частотам
#     память    — деформация связи тем, что совпало (русло)
#
#     θ̇ᵢₘ = ωᵢₘ + Σⱼₙ Wᵢⱼ S(ωⱼₙ − ωᵢₘ) sin(θⱼₙ − θᵢₘ)
#     ω̇ᵢₘ = Σⱼₙ Wᵢⱼ S(ωⱼₙ − ωᵢₘ) sin(θⱼₙ − θᵢₘ) − λ (ωᵢₘ − m ωᵢ₁) − γ (ωᵢₘ − ω₀ᵢₘ)
#     Ẇᵢⱼ = ε ( Σₚₙ S cos(θᵢₚ − θⱼₙ) / Σₚₙ S − Wᵢⱼ )
#     S(Δ) = 1 / (1 + (Δ/Π)²)
#
# При одной моде на вибрацию и Π = ∞ это в точности прежние уравнения:
# у вибрации была одна частота, отклик не зависел от совпадения.
#
# Запуск:
#
#     julia julia/kuramoto/basic.jl

using Printf
using Random
using Statistics

const N = 10        # сколько вибраций в среде
const DT = 0.01     # шаг описания, с

"""
    Medium

Среда — не набор объектов, а взаимодействие вибраций.
У каждой вибрации не одна частота, а несколько: аккорд.

- `omega0`   — свои частоты: `omega0[i, m]` — мода m вибрации i;
- `omega`    — частоты сейчас: это и есть состояние;
- `theta`    — фазы;
- `coupling` — связь между вибрациями: `coupling[i, j]` — насколько j тянет i;
- `band`     — полоса отклика Π: чем уже, тем строже резонанс;
- `tuning`   — λ: строй, жёсткость аккорда: держит обертоны в их отношении
              к основной частоте (`ωᵢₘ → m · ωᵢ₁`); при λ = 0 аккорд не держится;
- `memory`   — ε: как быстро связь деформируется тем, что через неё прошло;
- `damping`  — γ: как быстро частоты возвращаются к своим.
"""
mutable struct Medium
    omega0::Matrix{Float64}
    omega::Matrix{Float64}
    theta::Matrix{Float64}
    coupling::Matrix{Float64}
    band::Float64
    tuning::Float64
    memory::Float64
    damping::Float64
end

"""
    overlap(Δω, band)

Отклик по совпадению частот: S(Δ) = 1 / (1 + (Δ/Π)²).
При Δ = 0 отклик полный; дальше падает как 1/Δ². Это и есть резонанс:
среда отзывается на то, что звучит на её частоте, и почти не отзывается
на то, что звучит далеко.
"""
overlap(Δω::Float64, band::Float64) = 1 / (1 + (Δω / band)^2)

"""
    step!(medium, dt)

Один шаг описания: связь двигает частоты, затухание возвращает их к своим,
прошедшее деформирует связь.
"""
function step!(m::Medium, dt::Float64)
    vibs, modes = size(m.omega)
    drive = zeros(vibs, modes)
    for i in 1:vibs, p in 1:modes
        d = 0.0
        for j in 1:vibs
            j == i && continue
            for n in 1:modes
                s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
                d += m.coupling[i, j] * s * sin(m.theta[j, n] - m.theta[i, p])
            end
        end
        drive[i, p] = d
    end
    if m.tuning != 0.0
        for i in 1:vibs, p in 2:modes
            drive[i, p] -= m.tuning * (m.omega[i, p] - p * m.omega[i, 1])
        end
    end
    m.omega .+= dt .* (drive .- m.damping .* (m.omega .- m.omega0))
    m.theta .+= dt .* m.omega
    if m.memory != 0.0
        for i in 1:vibs, j in 1:vibs
            i == j && continue
            acc, weight = 0.0, 0.0
            for p in 1:modes, n in 1:modes
                s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
                acc += s * cos(m.theta[i, p] - m.theta[j, n])
                weight += s
            end
            target = weight == 0.0 ? m.coupling[i, j] : acc / weight
            m.coupling[i, j] += dt * m.memory * (target - m.coupling[i, j])
        end
    end
    return m
end

# --- что можно увидеть в описанном ---------------------------------------

"""
    coherence(medium)

Согласованность среды: звучит ли вся среда как одно.
Единица — все фазы совпали, ноль — фазы разошлись полностью.
"""
coherence(m::Medium) = abs(mean(exp.(im .* m.theta)))

"""
    partial_coherence(medium, p)

Согласованность по одной частоте: сошлись ли между собой p-е моды
разных вибраций. Именно здесь видно резонанс: среда может не сходиться
в одно целое, а сходиться по совпадениям частот.
"""
partial_coherence(m::Medium, p::Int) = abs(mean(exp.(im .* m.theta[:, p])))

"""
    chord(medium, i, t)

Что звучит i-я вибрация в момент t: её аккорд целиком, все моды сразу.
"""
chord(m::Medium, i::Int, t::Float64) = mean(cos.(m.omega[i, :] .* t .+ m.theta[i, :]))

"""
    wave(medium, t)

Что слышно в точке среды: наложение всех мод всех вибраций.
"""
wave(m::Medium, t::Float64) = mean(cos.(m.omega .* t .+ m.theta))

"""
    resonance(medium)

Разброс частот по всей среде: чем ближе частоты, тем уже резонанс.
"""
resonance(m::Medium) = maximum(m.omega) - minimum(m.omega)

"""
    spread(medium, i)

Разброс частот внутри одной вибрации: ширина её аккорда.
"""
spread(m::Medium, i::Int) = maximum(m.omega[i, :]) - minimum(m.omega[i, :])

"""
    tuning_error(medium, i)

Насколько строй i-й вибрации сошёл: отношение обертонов к основной частоте.
Ноль — аккорд держится ровно, как задан.
"""
function tuning_error(m::Medium, i::Int)
    modes = size(m.omega, 2)
    modes == 1 && return 0.0
    base = m.omega[i, 1]
    return maximum(abs(m.omega[i, p] - p * base) for p in 2:modes)
end

"""
    perturb!(medium, i, shift, p = 1)

Событие: толчок. Меняет частоту моды p вибрации i. Среда ничего об этом
не знает — она просто становится другой.
"""
perturb!(m::Medium, i::Int, shift::Float64, p::Int = 1) = (m.omega[i, p] += shift; m)

"""
    drift(medium, i, p = 1)

Отклонение: насколько частота моды ушла от своей.
"""
drift(m::Medium, i::Int, p::Int = 1) = abs(m.omega[i, p] - m.omega0[i, p])

# --- построение -----------------------------------------------------------

"""
    build(; modes = 1, link = 0.6, band = Inf, tuning = 0.0,
            memory = 1e-3, damping = 0.8, seed = 4)

Построить среду: N вибраций, у каждой modes частот.

При `modes = 1` и `band = Inf` это прежнее описание: одна частота на вибрацию,
отклик без различия совпадений. Аккорд, полоса и строй — тот же самый язык,
просто теперь в нём есть то, что слово называло аккордом.
"""
function build(; modes::Int = 1, link::Float64 = 0.6, band::Float64 = Inf,
                 tuning::Float64 = 0.0, memory::Float64 = 1e-3,
                 damping::Float64 = 0.8, seed::Int = 4)
    modes >= 1 || throw(ArgumentError("у вибрации хотя бы одна частота"))
    rng = MersenneTwister(seed)
    base = 1.0 .+ 0.15 .* randn(rng, N)          # основные частоты вибраций
    omega0 = zeros(N, modes)
    for i in 1:N
        omega0[i, 1] = base[i]
        for p in 2:modes
            omega0[i, p] = base[i] * p * (1 + 0.01 * randn(rng))   # обертоны
        end
    end
    theta = 2π .* rand(rng, N, modes)
    return Medium(omega0, copy(omega0), theta, fill(link / N, N, N),
                  band, tuning, memory, damping)
end

"""
    advance!(medium, seconds)

Прожить средой указанное время.
"""
function advance!(m::Medium, seconds::Float64)
    for _ in 1:round(Int, seconds / DT)
        step!(m, DT)
    end
    return m
end

"""
    show(medium, label)

Показать, в каком состоянии среда.
"""
function show(m::Medium, label::AbstractString)
    @printf("%-28s согласованность %.3f   разброс частот %.3f   связь %.4f\n",
            label, coherence(m), resonance(m), mean(m.coupling))
    modes = size(m.omega, 2)
    if modes > 1
        parts = join([@sprintf("%.3f", partial_coherence(m, p)) for p in 1:modes], "  ")
        @printf("%-28s по совпадениям:              %s\n", "", parts)
    end
    return nothing
end

# --- чтение ---------------------------------------------------------------

"""
    main()

Прочитать описание: сцена за сценой, без вопросов к природе.

1. Одна нота у каждой вибрации: среда сходится, наложение, толчок, память.
2. Аккорд: у вибрации три частоты; строй держит их в отношении,
   узкая полоса собирает среду не в одно целое, а по совпадениям частот.
"""
function main()
    @printf("десять вибраций, шаг описания %.2f с\n\n", DT)

    println("--- одна нота у каждой вибрации " * "-"^28)
    m = build()
    show(m, "сразу")
    advance!(m, 60.0)
    show(m, "через минуту")

    println("\nналожение: что слышно в одной точке среды")
    for t in (0.0, 0.5, 1.0, 2.0)
        @printf("   момент %4.1f с   %+.4f\n", t, wave(m, t))
    end

    println("\nзатухание: толчок и возвращение к своей частоте")
    perturb!(m, 1, 1.5)
    @printf("   после толчка         отклонение %.3f\n", drift(m, 1))
    advance!(m, 6.0)
    @printf("   через шесть секунд   отклонение %.3f\n", drift(m, 1))

    println("\nпамять: связь деформируется тем, что через неё прошло")
    untouched = build()
    routed = build()
    @printf("   связь[1,2] у нетронутой среды      %.6f\n", untouched.coupling[1, 2])
    for _ in 1:200
        perturb!(routed, 1, 1.5)
        advance!(routed, 6.0)
    end
    @printf("   связь[1,2] у слышавшей двести раз  %.6f\n", routed.coupling[1, 2])
    show(untouched, "нетронутая среда")
    show(routed, "среда с руслом")

    println("\nодно и то же событие в двух средах")
    perturb!(untouched, 1, 1.5)
    perturb!(routed, 1, 1.5)
    advance!(untouched, 6.0)
    advance!(routed, 6.0)
    @printf("   нетронутая: отклонение %.3f\n", drift(untouched, 1))
    @printf("   с руслом:   отклонение %.3f\n", drift(routed, 1))
    println("   русло — не запись о событии. Это среда, которая стала другой.")

    println("\n--- аккорд: у вибрации не одна частота, а три " * "-"^15)
    a = build(modes = 3, band = 0.1, tuning = 8.0)
    println("   свои частоты первой вибрации   " *
            join([@sprintf("%.2f", x) for x in a.omega0[1, :]], "  "))
    println("   звук аккорда в четырёх точках  " *
            join([@sprintf("%+.3f", chord(a, 1, t)) for t in (0.0, 0.5, 1.0, 2.0)], "  "))
    advance!(a, 60.0)
    println("   через минуту                   " *
            join([@sprintf("%.2f", x) for x in a.omega[1, :]], "  "))
    show(a, "среда из аккордов")
    @printf("   строй первой вибрации          %.4f\n", tuning_error(a, 1))
    println("   среда не сошлась в одно целое: три голоса живут на своих частотах.")

    println("\nтот же аккорд без строя: половина вибрации уезжает")
    b = build(modes = 3, band = 0.1, tuning = 0.0)
    advance!(b, 60.0)
    println("   через минуту                   " *
            join([@sprintf("%.2f", x) for x in b.omega[1, :]], "  "))
    @printf("   строй первой вибрации          %.4f\n", tuning_error(b, 1))
    println("   обертоны не следуют за основной: аккорд не одно целое.")

    println("\nто же с широкой связью: совпадения не различаются")
    c = build(modes = 3, band = Inf, tuning = 8.0)
    advance!(c, 60.0)
    show(c, "широкая связь")
    @printf("   строй первой вибрации          %.4f\n", tuning_error(c, 1))
    println("   полоса решает, что с чем сходится: узкая различает, широкая — нет.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

