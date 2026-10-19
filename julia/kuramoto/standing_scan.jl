# Entrain. Граница N: «стоячая доля» — доля или невязка равновесия?
#
# Сверка (`consistency.jl`) при r₀ = 0 проверяет
#
#     |λ·r/D − λ/(λ+γ)| ≤ 2e-2,     D = D_p − p·D₁.
#
# При N = 10 худшее расхождение 6.74e-3, при N = 30 — 2.44e-2. Вопрос: это
# ошибка меры, закон по N или граница описания.
#
# Алгебра сначала. Для промаха r = ω_p − p·ω₁ и r₀ = 0 уравнение шага даёт
#
#     dr/dt = D − λ·r − γ·(r − r₀) = D − (λ+γ)·r.
#
# Значит
#
#     q ≡ D − (λ+γ)·r = dr/dt,
#     e ≡ |λ·r/D − λ/(λ+γ)| = (λ/(λ+γ))·|q|/|D|.
#
# Равенство стоячей и скоростной долей — не отдельный закон, а условие q = 0:
# оно верно в стоячем состоянии и только там. Сверка берёт один срез через 60 с
# и не проверяет, что промах стоит. Если q ≠ 0, e — не расхождение двух картин,
# а скорость промаха, делённая на тягу.
#
# Критерий объявлен до прогона. Сканируются N = 10, 20, 30, 40; те же три
# расклада (λ,γ) = (16,0.8), (8,1), (4,4) и шесть зёрен, что в сверке; срез
# T = 60 с; для худшего случая каждого N — след до 1200 с. Считаются:
#   e_claim  — e в месте наибольшего промаха (как в сверке);
#   e_fixed  — e в том же месте (i,p), выбранном при T = 60, на всём следе;
#   e_med    — медиана e по всем местам (p > 1);
#   q_rel    — |q|/|D|, d_rel — |D|/(|D_p| + p·|D₁|).
#
#   1. Воспроизведение. При N = 10 max e_claim ≈ 6.74e-3, при N = 30 ≈ 2.44e-2.
#   2. Мера. Хотя бы при одном N e_claim(60) > 2e-2, но e_fixed падает с T
#      (e_fixed(600) ≤ 0.5·e_fixed(60) или e_fixed(600) ≤ 2e-2) и/или средняя
#      доля на позднем окне |λ·⟨r⟩/⟨D⟩ − λ/(λ+γ)| ≤ 2e-2. Тогда отказ — от
#      среза, а не от N; равенство надо писать для стоячего (q = 0) или для
#      средних по окну, а не для мгновенного среза.
#   3. Закон по N. Поздняя e_fixed(1200) > 2e-2 и растёт с N монотонно, не
#      меньше чем в 1.2 раза на каждом шаге, а q не убывает с T. Тогда у
#      невязки есть закон масштабирования, и его надо выводить.
#   4. Граница. e не убывает с T, но и не растёт монотонно с N; тогда стоячего
#      на этом окне нет, и равенство standing = speed — не закон, а условие
#      применимости, которое при больших N не выполнено. Это граница
#      утверждения, а не потеря какой-либо величины.
#
# Итог прогона 2026-10-19. Это мера. Худшее e(60): 6.74e-3, 3.10e-3,
# 2.44e-2, 4.84e-3 — немонотонно. При N = 30 e = s·q/D: q/D = 2.74e-2,
# d_rel = 0.976, значит D не мала; средняя доля на окне 600…1200 даёт 1.26e-3
# (худшее по N — 2.12e-3 при N = 20). e_fixed там же через 1200 с: 0.463
# (N = 20), 0.080 (N = 30) — стоячего среза нет. Порог 2e-2 остаётся для
# q = 0 или для средних по окну.
#
# Запуск (по одному N на процесс: N — const в basic.jl):
#
#     ENTRAIN_N=30 julia julia/kuramoto/standing_scan.jl

include(joinpath(@__DIR__, "excess.jl"))

const SEEDS = 1:6
const WRITES = ((16.0, 0.8), (8.0, 1.0), (4.0, 4.0))
const T0 = 60.0
const SNAPS = (60.0, 120.0, 300.0, 600.0, 1200.0)
const SAMPLE = 1.0

"""
    exact_chord(; seed, tuning, damping)

Тот же точный аккорд, что в `consistency.jl`: ω₀_p = p·ω₀₁, затем ω = ω₀.
"""
function exact_chord(; seed::Int, tuning::Float64, damping::Float64)
    m = build(modes = 3, band = 0.1, tuning = tuning, damping = damping, seed = seed)
    for i in 1:size(m.omega0, 1), p in 2:size(m.omega0, 2)
        m.omega0[i, p] = p * m.omega0[i, 1]
    end
    m.omega .= m.omega0
    return m
end

"Место наибольшего относительного промаха — как `miss_place` в сверке."
function miss_place(m::Medium)
    worst, at = 0.0, (1, 1)
    for i in axes(m.omega, 1), p in 2:size(m.omega, 2)
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        x = abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base)
        x > worst && ((worst, at) = (x, (i, p)))
    end
    return at
end

"""
    terms_at(m, i, p, lam, gamma)

r, D, q = D − (λ+γ)r (при r₀ = 0), e = |λr/D − λ/(λ+γ)|, q_rel и d_rel.
`dscale = |D_p| + p·|D₁|` — масштаб самой тяги: d_rel показывает, не является
ли D почти сокращением двух больших членов.
"""
function terms_at(m::Medium, i::Int, p::Int, lam::Float64, gamma::Float64)
    r0 = written(m, i, p)
    r = m.omega[i, p] - p * m.omega[i, 1]
    dp, d1 = drag(m, i, p), drag(m, i, 1)
    d = dp - p * d1
    q = d - lam * r - gamma * (r - r0)
    s = lam / (lam + gamma)
    base = abs(d)
    e = base > 1e-14 ? abs(lam * r / d - s) : NaN
    qrel = base > 1e-14 ? abs(q) / base : NaN
    dscale = abs(dp) + p * abs(d1)
    drel = dscale > 0.0 ? base / dscale : NaN
    ident = isfinite(e) ? abs(e - s * abs(q) / base) : NaN
    return (r = r, d = d, q = q, e = e, qrel = qrel, drel = drel, ident = ident,
            r0 = r0, dp = dp, d1 = d1)
end

"Статистика по всем местам (p > 1): медиана и максимум e, q_rel и d_rel."
function all_places(m::Medium, lam::Float64, gamma::Float64)
    es, qs, ds = Float64[], Float64[], Float64[]
    for i in axes(m.omega, 1), p in 2:size(m.omega, 2)
        t = terms_at(m, i, p, lam, gamma)
        isfinite(t.e) || continue
        push!(es, t.e); push!(qs, t.qrel); push!(ds, t.drel)
    end
    isempty(es) && return (e_med = NaN, e_max = NaN, q_med = NaN, d_med = NaN)
    return (e_med = median(es), e_max = maximum(es),
            q_med = median(qs), d_med = median(ds))
end

function scan_cross()
    rows = NamedTuple[]
    println("N = $N, срез T = $(T0) с, точный аккорд r₀ = 0")
    println("  λ     γ  зерно  место        e_claim     q_rel      d_rel      e_med      e_max")
    for (lam, gamma) in WRITES, seed in SEEDS
        m = exact_chord(seed = seed, tuning = lam, damping = gamma)
        advance!(m, T0)
        i, p = miss_place(m)
        t = terms_at(m, i, p, lam, gamma)
        a = all_places(m, lam, gamma)
        push!(rows, (lam = lam, gamma = gamma, seed = seed, i = i, p = p,
                     e = t.e, qrel = t.qrel, drel = t.drel, r = t.r, d = t.d,
                     q = t.q, ident = t.ident, e_med = a.e_med, e_max = a.e_max,
                     q_med = a.q_med, d_med = a.d_med))
        @printf("  %4.1f %5.1f  %4d  (%2d,%d)  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                lam, gamma, seed, i, p, t.e, t.qrel, t.drel, a.e_med, a.e_max)
    end
    worst = rows[argmax([r.e for r in rows])]
    @printf("  худшее: λ = %.1f, γ = %.1f, зерно %d, место (%d,%d): e = %.4e, q_rel = %.4e, d_rel = %.4e\n",
            worst.lam, worst.gamma, worst.seed, worst.i, worst.p, worst.e,
            worst.qrel, worst.drel)
    @printf("  тождество e = s·|q|/|D|: max ошибка %.2e\n",
            maximum(r.ident for r in rows))
    return rows, worst
end

"""
    trace_case(row)

Продолжить худший случай до 1200 с. На каждом срезе печатаются e в месте
наибольшего промаха, e в месте, выбранном при 60 с, медиана по местам и
средняя доля λ⟨r⟩/⟨D⟩ на позднем окне [600,1200].
"""
function trace_case(row::NamedTuple)
    m = exact_chord(seed = row.seed, tuning = row.lam, damping = row.gamma)
    advance!(m, T0)
    i0, p0 = miss_place(m)
    late_r = late_d = late_q = late_e = late_qrel = 0.0
    late_n = 0
    last = T0
    println()
    @printf("  продольный след: λ = %.1f, γ = %.1f, зерно %d, место (%d,%d)\n",
            row.lam, row.gamma, row.seed, i0, p0)
    println("      T,с   e_claim    q_rel     e_fixed   q_fixed   e_med     e_max    ⟨e_fixed⟩  ⟨q/D⟩    λ⟨r⟩/⟨D⟩")
    for T in SNAPS
        T <= T0 + 1e-9 && continue
        t = last
        sum_e = 0.0
        n_sum = 0
        while t < T - 1e-9
            dt = min(SAMPLE, T - t)
            advance!(m, dt)
            t += dt
            ft = terms_at(m, i0, p0, row.lam, row.gamma)
            if isfinite(ft.e)
                sum_e += ft.e
                n_sum += 1
                if t >= 600.0 - 1e-9
                    late_r += ft.r; late_d += ft.d; late_q += ft.q
                    late_e += ft.e; late_qrel += ft.qrel; late_n += 1
                end
            end
        end
        last = T
        i, p = miss_place(m)
        c = terms_at(m, i, p, row.lam, row.gamma)
        f = terms_at(m, i0, p0, row.lam, row.gamma)
        a = all_places(m, row.lam, row.gamma)
        mean_e = n_sum == 0 ? NaN : sum_e / n_sum
        mean_qd = late_d == 0.0 ? NaN : abs(late_q / late_d)
        mean_share = late_d == 0.0 ? NaN :
                     abs(row.lam * late_r / late_d - row.lam / (row.lam + row.gamma))
        @printf("  %8.0f  %9.3e  %9.3e  %9.3e  %9.3e  %9.3e  %9.3e  %10.3e  %9.3e  %10.3e\n",
                T, c.e, c.qrel, f.e, f.qrel, a.e_med, a.e_max, mean_e, mean_qd, mean_share)
    end
    late_share = late_d == 0.0 ? NaN :
                 abs(row.lam * late_r / late_d - row.lam / (row.lam + row.gamma))
    late_qd = late_d == 0.0 ? NaN : abs(late_q / late_d)
    late_me = late_n == 0 ? NaN : late_e / late_n
    println("  позднее окно 600…1200 с: |λ⟨r⟩/⟨D⟩ − λ/(λ+γ)| = " * @sprintf("%.3e", late_share) *
            ", среднее |q|/|D| = " * @sprintf("%.3e", late_qd) *
            ", среднее e = " * @sprintf("%.3e", late_me) *
            ", срезов " * string(late_n))
    return nothing
end

function main()
    println("Стоячая доля: доля формы или скорость промаха?")
    println()
    rows, worst = scan_cross()
    trace_case(worst)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
