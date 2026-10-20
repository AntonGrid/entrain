# Entrain. Горизонт N: почему link/N — не потеря связи, а среднее поле.
#
# Вопрос: при N = 1 000 000 связь на пару link/N ~ 6e-7, меньше разброса
# собственных частот (0.15). Распадается ли среда? Здесь тот же закон читается
# при N = 100 и 200; миллион не прогоняется, а считается формулой.
#
# Обозначения:
#   link = K — полная тяга на вибрацию (сумма W_ij по соседям), не связь на пару;
#   W_ij = K/N — связь на пару в полном графе, K/z — в городе;
#   R — согласованность (coherence);
#   F_pair = K/N;
#   F_coh = N·⟨W⟩·R = K·R   — если фазы складываются в фазе;
#   F_inc ~ √N·(K/N) = K/√N  — если фазы складываются случайно.
#
# Для одной моды, band = Inf, без памяти, в стоячем состоянии:
#   F_i = K·R·sin(ψ − θ_i),   затухание: F_i = γ(Ω − ω0_i),
#   Kc = 2γ/(π·g(0)) = 2γσ√(2π)/π,
#   R = ∫_{|x|≤KR/γ} √(1 − (γx/(KR))²)·g(x) dx.
# При γ = 0.8, σ = 0.15: Kc ≈ 0.1915, R∞(K = 0.6) ≈ 0.9784.
#
# Проверка: при N = 100 и 200, link = 0.6, R = 0.978 и 0.979, а сила на пару
# падает с 0.006 до 0.003. Это не закон по N: R сходится к R∞, а не падает.
#
# Память: при ε > 0 цель W_ij → (link + N₀·cos(θ_i − θ_j))/K, где K — число
# соседей (N в полном графе, z в городе), N₀ — опорное число вибраций памяти.
# Поэтому к сроку 1/ε связь на пару остаётся O(1/K), а полная тяга K·⟨W⟩ —
# O(1), не K. Предел полной тяги — link + N₀·⟨cos⟩: память помнит и старт
# среды, и совпадение фаз. Проверено при N = 10, 100 и в городе N = 100, z = 4:
# Ktot стоит, пара меняется как N/K, R — свойство графа.
#
# Запуск:
#
#     julia julia/kuramoto/horizon.jl          # N = 100
#     ENTRAIN_N=200 julia julia/kuramoto/horizon.jl
#     ENTRAIN_N=10  julia julia/kuramoto/horizon.jl

ENV["ENTRAIN_N"] = get(ENV, "ENTRAIN_N", "100")

include(joinpath(@__DIR__, "basic.jl"))

using Statistics

const SIGMA = 0.15
const GAMMA = 0.8
const KAPPAS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.60)
const SEEDS = 1:4
const T = 60.0

"""
    mean_link(medium)

Средняя связь по всем связям, кроме собственных: то, что видит одна вибрация
от одной соседки. У свежей среды это link/N.
"""
function mean_link(m::Medium)
    s, n = 0.0, 0
    for i in axes(m.coupling, 1), j in axes(m.coupling, 2)
        i == j && continue
        m.coupling[i, j] == 0.0 && continue     # в городе это не сосед
        s += m.coupling[i, j]
        n += 1
    end
    return n == 0 ? 0.0 : s / n
end

"""
    critical_link(; sigma, gamma)

Порог полной тяги для одной моды, широкой полосы и стоячего состояния:

    Kc = 2γ/(π·g(0)) = 2γσ√(2π)/π.

Это тяга на вибрацию, а не на пару; от N она не зависит.
"""
critical_link(; sigma::Float64 = SIGMA, gamma::Float64 = GAMMA) =
    2 * gamma * sigma * sqrt(2π) / π

"""
    order_inf(link; sigma, gamma)

Согласованность в термодинамическом пределе: корень

    R = ∫_{|x|≤link·R/γ} √(1 − (γx/(link·R))²)·g(x) dx,

где g — плотность нормального разброса своих частот.
"""
function order_inf(link::Float64; sigma::Float64 = SIGMA,
                    gamma::Float64 = GAMMA, n::Int = 4001,
                    iterations::Int = 2000)
    R = 1.0
    for _ in 1:iterations
        a = link * R / gamma
        a <= 0.0 && return 0.0
        s = 0.0
        for k in 1:n
            x = -a + (k - 0.5) * 2a / n
            s += sqrt(max(0.0, 1 - (x / a)^2)) * exp(-x^2 / (2sigma^2))
        end
        new = s * (2a / n) / (sigma * sqrt(2π))
        abs(new - R) <= 1e-12 && return new
        R = new
    end
    return R
end

"""
    case(link; seed, memory, seconds)

Один прогон: среда с одной модой и широкой полосой. Возвращает согласованность,
связь на пару, полную тягу на вибрацию K·⟨W⟩ (K = N в полном графе, z в городе)
и коллективную силу K·⟨W⟩·R.
"""
function case(link::Float64; seed::Int, memory::Float64 = 0.0,
              seconds::Float64 = T)
    m = build(modes = 1, band = Inf, link = link, memory = memory, seed = seed)
    advance!(m, seconds)
    w = mean_link(m)
    R = coherence(m)
    return (R = R, Wbar = w, Ktot = m.degree * w, pair = link / m.degree, coll = m.degree * w * R)
end

function main()
    K = (0 < DEGREE < N) ? DEGREE : N
    model = K == N ? "полный граф" : "город z = $K"
    println("Горизонт N: link/K — среднее поле, а не потеря связи; $model")
    @printf("N = %d, K = %d, T = %.0f с, зёрна %d…%d, одна мода, полоса Inf\n",
            N, K, T, first(SEEDS), last(SEEDS))
    @printf("Kc = 2γσ√(2π)/π = %.4f при γ = %.1f, σ = %.2f\n",
            critical_link(), GAMMA, SIGMA)
    println()
    println("  link    R_med    R_min    R_max      R∞     pair      coll")
    for link in KAPPAS
        v = [case(link; seed = s) for s in SEEDS]
        R = sort([x.R for x in v])
        w = median([x.Wbar for x in v])
        @printf("  %4.2f  %8.4f  %8.4f  %8.4f  %8.4f  %8.2e  %8.4f\n",
                link, median(R), minimum(R), maximum(R), order_inf(link),
                link / K, K * w * median(R))
    end
    println()
    println("  память ε = 1e-3, link = 0.60, цель (link+N₀·cos)/K: Ktot = O(1), pair = O(1/K)")
    for s in SEEDS
        x = case(0.60; seed = s, memory = 1e-3)
        @printf("   зерно %d: R = %.4f  Wbar = %.3e  Ktot = %.3f  coll = %.3f\n",
                s, x.R, x.Wbar, x.Ktot, x.coll)
    end
    println()
    println("  При link = 0.60 и N = 1e6 в полном графе: без памяти pair = 6.0e-7;")
    println("  память с целью (link+N₀·cos)/K держит Ktot = link+N₀·R конечным, а не K.")
    println("  В городе при z = O(1) pair = link/z не падает с N, а Ktot — то же O(1).")
    println("  В полном графе масштаб один: W_ij = link/N; при степени z — W_ij = link/z;")
    println("  память города нормируется степенью: цель (link + N₀·cos)/z.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

