# Entrain. Курамото: потолок тяги по записи — из чего следует малость перевеса.
#
# Пункт 4 реестра открытого требует доказать F ≤ 2.0 (малость перевеса) или
# признать малость первичной. Здесь проверяется, следует ли она из записи.
#
# Грубый потолок (`|D_ip| ≤ Σ c_ij·N_m`) считает |sin| ≤ 1 и overlap ≤ 1 — и
# даёт 1.62, откуда F ≈ 2.44 и достаточная жёсткость 243 выше потолка шага
# (199.2). Но overlap в записи — не «≤ 1», а точное ядро
#
#     overlap(Δ, band) = 1 / (1 + (Δ/band)²),   band = 0.1,
#
# а основные частоты разбросаны как 1.0 ± 0.15 (`build`). Значит в строку
# (i, p) попадают не все девять чужих вибраций, а только те, чья гармоника
# стоит ближе полосы: чужие гармоники отсечены ядром. Потолок, посчитанный
# по этому ядру, — не оценка «на глаз», а точная верхняя граница при |sin| ≤ 1.
#
# Порог объявлен до прогона: малость следует из записи, если у всех шести
# зёрен и обеих строк p = 2, 3
#
#     Fcap = (Dcap_p + p·Dcap_1 + γ·|r₀_p|) / (p·ω₁) ≤ 2.0,
#
# где Dcap_p = Σ_{j≠i} c_ij·Σ_n overlap(ω_jn − ω_ip, band) — потолок тяги
# строки; c_ij — фактические, снятые через 60 с держания (память 1e-3 их
# двигает, и потолок нужен для того состояния, в котором корень ищется);
# r₀_p — написанный промах. Не так хотя бы у одного зерна — малость не
# следует, и её надо принять аксиомой.
#
# Итог (тридцать первая запись): потолок по ядру — 0.745 … 1.147 при планке
# 2.0, запас у худшего зерна 1.7 раза, достаточная жёсткость 113.9 ниже
# потолка шага 199.2. Малость следует из записи, и держат её два числа:
# полоса 0.1 и разброс основных частот 0.15 — чужие гармоники отсечены ядром.
#
# Запуск:
#
#     julia julia/kuramoto/ceiling.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6
const ROOTS = (40.83, 44.64, 48.93, 52.31, 53.56, 65.37)  # корни из `self.jl`, окно 60 с
const BAND = 0.1
const LINK = 0.6
const GAMMA = 0.8
const WINDOW = 60.0
const NEED = 0.01
const PLANKA = 2.0        # F ≤ s·(λ_max + γ) при s = 1 %, λ_max = 199.2

"Промах обертонов: наибольшее |ω_p − p·ω₁|/(p·ω₁) по всем вибрациям и обертонам."
function miss_of(m::Medium)
    worst = 0.0
    for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        worst = max(worst, abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base))
    end
    return worst
end

"""
    ceiling(seed, lam)

Среда того же вида, что в `self.jl`, проживает минуту при своей жёсткости; с неё
снимаются фактические связи и потолок тяги по ядру перекрытия. Возвращается
наибольшая F по строкам p = 2, 3 и место, где она стоит.
"""
function ceiling(seed::Int, lam::Float64)
    m = build(modes = 3, band = BAND, link = LINK, tuning = lam,
              damping = GAMMA, memory = 1e-3, seed = seed)
    for _ in 1:round(Int, WINDOW / DT)
        step!(m, DT)
        (all(isfinite, m.omega) && all(isfinite, m.theta)) || return nothing
    end
    vibs, modes = size(m.omega)
    cap = zeros(vibs, modes)
    for i in 1:vibs, p in 1:modes
        s = 0.0
        for j in 1:vibs
            j == i && continue
            for n in 1:modes
                s += m.coupling[i, j] * overlap(m.omega[j, n] - m.omega[i, p], m.band)
            end
        end
        cap[i, p] = s
    end
    best, at = -Inf, (1, 1)
    for i in 1:vibs, p in 2:modes
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        r0 = m.omega0[i, p] - p * m.omega0[i, 1]
        f = (cap[i, p] + p * cap[i, 1] + GAMMA * abs(r0)) / (p * base)
        f > best && ((best, at) = (f, (i, p)))
    end
    return (fcap = best, at = at, miss = miss_of(m), cmax = maximum(m.coupling),
            dcap = maximum(cap))
end

function main()
    println("Потолок тяги по записи: следует ли из неё малость перевеса?")
    println("Среда та же, что в `self.jl`: три моды, связь 0.6, полоса 0.1, затухание")
    println("0.8, память 1e-3; шесть зёрен, шаг 0.01; окно — минута, как у корней.")
    println("Потолок считается по ядру 1/(1+(Δ/band)²), а не по «overlap ≤ 1».")
    @printf("Планка до прогона: Fcap ≤ %.1f у всех шести зёрен и строк p = 2, 3.\n", PLANKA)

    println("\n--- потолок по шести зёрнам " * "-"^10)
    println("  зерно     λ     промах    max c_ij   max Dcap   Fcap   планка/Fcap")
    fcaps = Float64[]
    worst = 0.0
    for (k, s) in enumerate(SEEDS)
        r = ceiling(s, ROOTS[k])
        if r === nothing
            println(@sprintf("  %5d  %6.2f   срыв", s, ROOTS[k]))
            push!(fcaps, Inf)
            continue
        end
        push!(fcaps, r.fcap)
        worst = max(worst, r.fcap)
        println(@sprintf("  %5d  %6.2f   %6.3f %%   %7.3f   %8.3f   %6.3f   %5.1f",
                         s, ROOTS[k], 100 * r.miss, r.cmax, r.dcap, r.fcap,
                         PLANKA / r.fcap))
    end

    println("\n--- итог " * "-"^21)
    @printf("Fcap по зёрнам: %7.3f … %-7.3f (наибольшее — %.3f)\n",
            minimum(fcaps), maximum(fcaps), worst)
    @printf("Планка %.1f; запас у худшего зерна — %.1f раза.\n", PLANKA, PLANKA / worst)

    if worst <= PLANKA
        println("Малость следует из записи, и источник назван: ядро перекрытия")
        println("1/(1+(Δ/0.1)²) при разбросе основных частот 0.15 отсекает чужие")
        println("гармоники — тяга строки не 1.62, а доли единицы. Доказательство")
        println("существования корня этим закрыто: F ≤ планки при жёсткости у потолка")
        println("шага, и достаточная жёсткость остаётся ниже 199.2.")
    else
        println("Малость из записи не следует: Fcap выше планки. Тогда она")
        println("принимается аксиомой — наравне с первичностью перевеса.")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
