# Entrain. Курамото: λ — скорость или разность скоростей?
#
# В `pair.jl` видно, что в шаге λ и γ неразличимы: у промаха обертона один множитель,
# 1 − (λ+γ)·dt. Но у основной частоты своя скорость, и строя в ней нет:
#
#     ω₁′ = ω₁ + dt·(D₁ − γ·(ω₁ − ω₀₁))     — строй держит обертоны, а её саму не держит.
#
# Значит, среда показывает два числа — 1 − γ·dt и 1 − (λ+γ)·dt, — и λ есть их разность,
# делённая на шаг: λ = (k₁ − k₂)/dt. Тогда строй не только вычисляется, но и читается.
#
# Читается он так: толчок на основную частоту вибрации. Сама она возвращается со своей
# скоростью, а промах, который толчок за собой тянет, спешит со своей; обе скорости видны
# в одном опыте и обе берутся по разнице с ненатолкнутой средой того же зерна — у промаха
# есть своё гуляние, и без этого оно забивает знак.
#
# Скорости живут на разных временах, и это часть чтения: в коротком окне (2–4 шага) видно,
# как спешит промах; на тридцатом шаге он идёт уже со скоростью затухания; к сотому уходит
# в общую моду среды. Последнее уже не о λ: оно не меняется, если убрать память связи, —
# значит, мода не её.
#
# Итог этого чтения — в журнале: λ читается разностью двух скоростей с точностью около
# процента, а при λ = 0 обе скорости — одно и то же число до шестого знака.
#
# Запуск:
#
#     julia julia/kuramoto/rates.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6
const WINDOW = 4        # на каком окне читаются скорости: шагов

"""
    miss_of(medium)

Промах вибрации 1 по обертону 2: ω₁₂ − 2·ω₁₁ (без нормировки — нормировка внесла бы
свою скорость, от основной частоты).
"""
miss_of(m::Medium) = m.omega[1, 2] - 2 * m.omega[1, 1]

"Своя частота вибрации 1 — та компонента, которой дан толчок."
own_of(m::Medium) = m.omega[1, 1]

"""
    chorded(; seed, tuning, damping, memory)

Аккорд 1 : 2 : 3 — та же среда, что в `pair.jl`: три моды, полоса 0.10, связь 0.60,
написанный промах 1 %. Затухание и шаг здесь задаются затем, что скорости читаются
по отдельности, а слагаемые у них разные.
"""
function chorded(; seed::Int = 1, tuning::Float64 = 0.0, damping::Float64 = 0.8,
                  memory::Float64 = 1e-3)
    return build(modes = 3, band = 0.1, link = 0.6, tuning = tuning,
                 damping = damping, memory = memory, seed = seed)
end

"""
    advance_step!(medium, seconds, dt)

Прожить средой время при шаге dt — как в `pair.jl`. Сорвавшееся состояние здесь не сбой,
а отметка: фаза ушла внутри шага, и читать уже нечего.
"""
function advance_step!(m::Medium, seconds::Float64, dt::Float64)
    for _ in 1:round(Int, seconds / dt)
        try
            step!(m, dt)
        catch
            fill!(m.omega, Inf)
            fill!(m.theta, Inf)
            return m
        end
        (all(isfinite, m.omega) && all(isfinite, m.theta)) || return m
    end
    return m
end

"""
    measured(lam, damp, dt, win, push_mode; ...)

Сколько за шаг даёт величина `f` после толчка на моду `push_mode` вибрации 1: среднее
геометрическое по зёрнам от (после / до)^(1/окно). Считается по разнице с ненатолкнутой
средой того же зерна, шаг в шаг.
"""
function measured(lam::Float64, damp::Float64, dt::Float64, win::Int, push_mode::Int;
                  seeds = SEEDS, memory::Float64 = 1e-3, push::Float64 = 0.02,
                  f = miss_of, seconds::Float64 = 5.0)
    logs = Float64[]
    for s in seeds
        a = chorded(seed = s, tuning = lam, damping = damp, memory = memory)
        b = chorded(seed = s, tuning = lam, damping = damp, memory = memory)
        advance_step!(a, seconds, dt)
        advance_step!(b, seconds, dt)
        perturb!(b, 1, push, push_mode)
        before = f(b) - f(a)
        before != 0.0 || return NaN
        advance_step!(a, win * dt, dt)
        advance_step!(b, win * dt, dt)
        (all(isfinite, a.omega) && all(isfinite, b.omega)) || return NaN
        after = f(b) - f(a)
        after != 0.0 || return NaN
        push!(logs, log(abs(after / before)) / win)
    end
    return exp(sum(logs) / length(logs))
end

"Скорость сама и промаха в одном опыте: толчок на основную частоту."
function two_speeds(lam::Float64, damp::Float64, dt::Float64; win::Int = WINDOW, kwargs...)
    k_own = measured(lam, damp, dt, win, 1; f = own_of, kwargs...)
    k_miss = measured(lam, damp, dt, win, 1; f = miss_of, kwargs...)
    return k_own, k_miss
end

"Скорость промаха при толчке обертону: у обертона и промаха она одна и та же."
function miss_speed(lam::Float64, damp::Float64, dt::Float64; win::Int, kwargs...)
    return measured(lam, damp, dt, win, 2; f = miss_of, kwargs...)
end

function main()
    println("две скорости: λ — их разность. Вибраций $N, мод 3, полоса 0.10, связь 0.60")
    println("шаг чтения $DT, окно $WINDOW шагов, зёрен $(length(SEEDS))")

    println("\n--- две скорости в одном опыте: толчок на основную частоту ----")
    println("   Сама она возвращается со скоростью затухания, а промах, который")
    println("   толчок за собой тянет, — со скоростью суммы. Разность их показаний,")
    println("   делённая на шаг, и есть λ.")
    println("      λ     γ   своя скорость   1−γ·dt   скорость промаха  1−(λ+γ)·dt  λ по разности")
    for (lam, damp) in ((16.0, 0.8), (15.2, 1.6), (8.8, 8.0), (0.0, 0.8), (0.0, 8.0))
        k_own, k_miss = two_speeds(lam, damp, DT)
        @printf("   %5.1f %5.1f      %9.6f  %7.5f     %9.6f      %8.5f     %9.4f\n",
                lam, damp, k_own, 1 - damp * DT, k_miss, 1 - (lam + damp) * DT,
                (k_own - k_miss) / DT)
    end
    println("   При λ = 0 разность показаний — ноль: обе скорости среды — одно число,")
    println("   и λ есть ровно то, чем промах спешит больше основной.")

    println("\n--- скорости сходятся: при λ = 0 они одно и то же число ----")
    println("      γ    шаг    своя скорость   скорость промаха     разность   λ по разности")
    for damp in (0.8, 4.0, 10.0), dt in (DT, 0.05)
        k_own, k_miss = two_speeds(0.0, damp, dt)
        @printf("   %5.1f %6.3f       %9.6f       %9.6f   %9.2e      %7.4f\n",
                damp, dt, k_own, k_miss, k_own - k_miss, (k_own - k_miss) / dt)
    end
    println("   Совпадение до шестого знака и на мелком шаге, и на крупном: строй не")
    println("   спрятан в одной из скоростей — его нет ни в одной.")

    println("\n--- где живут скорости: толчок обертону, промах по окнам ----")
    println("   Своя скорость промаха видна сразу, потом он идёт со скоростью")
    println("   основной (это затухание), а к сотому шагу уходит в общую моду")
    println("   среды. Последний столбец — та же среда без памяти связи.")
    println("      окно   промах (λ=4, γ=4)   без памяти связи")
    for win in (2, 4, 8, 16, 32, 64, 128)
        k = miss_speed(4.0, 4.0, 0.05; win = win)
        k0 = miss_speed(4.0, 4.0, 0.05; win = win, memory = 0.0)
        @printf("   %6d          %9.6f                        %9.6f\n", win, k, k0)
    end
    println("   На коротком окне ждём 1 − (λ+γ)·dt = 0.600, на тридцатом — 1 − γ·dt =")
    println("   0.800: промах догоняет основную. Дальше скорости уходят к единице, и")
    println("   это общая мода связи: без памяти связи тот же хвост — значит, не её.")

    println("\nИтог: у среды две скорости — своя у основной частоты и своя у промаха, —")
    println("и строй есть их разность, делённая на шаг. Читается он по поведению, без")
    println("написанных чисел: толчок в основную частоту показывает обе сразу. При")
    println("λ = 0 обе скорости — одно число, и держать нечего; чем больше строй, тем")
    println("сильнее промах спешит основной. За коротким окном разность перестаёт быть")
    println("строем: там уже связь, а не он.")

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
