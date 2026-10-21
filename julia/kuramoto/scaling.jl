# Entrain. Граница N: почему «место записано» ломается при N = 30.
#
# В сверке при N = 10 место записано: отличие у места события больше, чем вдали
# (у 1-й вибрации 21.5, у 5-й 7.1). При N = 30 те же два пункта дали 4.09 и 0.57
# при пороге 5 — на цели памяти `link + N₀·cos` без деления на K. После коммита
# 460bbe4 цель стала `(link + N₀·cos)/K`, и тот же прогон при N = 30 даёт 8.79
# (2026-10-20): промах был не в месте, а в цели, которую среда писала. Сцена
# остаётся чтением меры: как near/far ведёт себя по N, если цель не менять.
# Вопрос ниже — про меру (near/far, где far — среднее по всем остальным связям,
# а их число растёт как N²), а не про саму запись места.
#
# Читается одно и то же прожитое при четырёх N. Для каждого N и места k:
#   near — среднее |δ| по связям, касающимся k (их 2(N−1));
#   far  — среднее |δ| по остальным связям (их (N−1)(N−2));
#   похожесть рельефов — отдельно по ближним и по дальним связям.
#
# Критерий объявлен до прогона: N = 10, 20, 30, 40; шесть зёрен; ε = 1e-3;
# 200 проходов по 6 с; места 1 и 5.
#
#   масштабирование меры — near не падает (near(40)/near(10) > 0.5), far растёт
#   с N или отношение падает степенным законом, похожесть ближних связей ≥ 0.5;
#
#   фундаментальный слом — near падает больше чем вдвое и/или похожесть ближних
#   связей ниже 0.5: место перестаёт записываться, и порог тут ни при чём.
#
# Запуск (по одному N на процесс: N — const в basic.jl):
#
#     ENTRAIN_N=30 julia julia/kuramoto/scaling.jl

include(joinpath(@__DIR__, "relief.jl"))

"""
    cor_subset(a, b, k; near)

Похожесть двух рельефов только по связям места `k` (`near = true`) или только
по связям вдали от него (`near = false`). Та же связь Пирсона, что в `agreement`,
но без смешения ближнего рельефа с дальним фоном.
"""
function cor_subset(a::AbstractMatrix{Float64}, b::AbstractMatrix{Float64}, k::Int;
                    near::Bool)
    va, vb = Float64[], Float64[]
    for i in axes(a, 1), j in axes(a, 2)
        i == j && continue
        ((i == k) || (j == k)) == near || continue
        push!(va, a[i, j])
        push!(vb, b[i, j])
    end
    return cor(va, vb)
end

function main()
    println("Граница N: место записано или тонет в дальнем фоне")
    println("N = $N, зёрен $(length(SEEDS)), проходов $PASSES по $PASS с, ε = $WIDE")
    println()
    println("  место  near_med  near_min   far_med   far_min  ratio_med  ratio_min  " *
            "похожесть: все   ближние   дальние")
    for k in (1, 5)
        near, far, ratio = Float64[], Float64[], Float64[]
        δs = Matrix{Float64}[]
        quiet_link, event_link = Float64[], Float64[]
        for s in SEEDS
            q = life(WIDE, PASSES; seed = s, events = false)
            r = life(WIDE, PASSES; seed = s, events = true, place = k)
            δ = relief_of(r, q)
            n, f = relief_place(δ, k)
            push!(near, n)
            push!(far, f)
            push!(ratio, n / f)
            push!(δs, δ)
            push!(quiet_link, mean_link(q))
            push!(event_link, mean_link(r))
        end
        pairs = [(a, b) for a in 1:length(δs) for b in (a + 1):length(δs)]
        all_agree = minimum(agreement(δs[a], δs[b]) for (a, b) in pairs)
        near_agree = minimum(cor_subset(δs[a], δs[b], k; near = true) for (a, b) in pairs)
        far_agree = minimum(cor_subset(δs[a], δs[b], k; near = false) for (a, b) in pairs)
        @printf("  %5d  %8.4f  %8.4f  %8.4f  %8.4f  %9.3f  %9.3f  %+9.3f  %+8.3f  %+8.3f\n",
                k, median_of(near), minimum(near), median_of(far), minimum(far),
                median_of(ratio), minimum(ratio), all_agree, near_agree, far_agree)
        @printf("         связь: без событий %.4f … %.4f, с событием %.4f … %.4f\n",
                minimum(quiet_link), maximum(quiet_link),
                minimum(event_link), maximum(event_link))
    end
    println()
    println("  far — среднее по всем связям вне места; их число растёт как N²,")
    println("  near — по 2(N−1) связям места. Похожесть ближних и дальних считается")
    println("  отдельно: иначе растущий дальний фон решает за рельеф места.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
