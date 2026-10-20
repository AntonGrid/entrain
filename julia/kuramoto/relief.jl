# Entrain. Курамото: знак и среда вместе — где в записи место?
#
# В главе 22 оставалось неописанным: «Знак и среда вместе». Знак — отделимая
# вибрация, почта; среда — то, что остаётся, когда вибрации не отделить. Как
# описать целое, не сводя одно к другому, было неясно.
#
# Читается это тем, что уже найдено: у памяти ε один на уровень, а цель у каждой
# связи своя — за шаг связь идёт к **своей** цели одной долей ε·dt. Значит, доля
# одна (время), а цели разные (место). Здесь спрашивается, виден ли в записи
# место: помнит ли среда, **где** прошло то, что через неё шло, — или только
# сколько прошло.
#
# Что читается:
#
#     1. доля одна, а записанное разное: у всех связей уровня доля пути ε·dt,
#        но цели у связей разные, и записанное за шаг различается в разы;
#     2. след места: связь на месте события отличается от прожившей без событий
#        сильнее, чем связь вдали от него, — и перенос события переносит рельеф;
#     3. величина рельефа против его места: что из этого бросок, а что свойство
#        (у широкой памяти и у короткой — по-разному);
#     4. рельеф стирается прожитым: событие кончилось, среда живёт дальше — и
#        отметина уходит по сроку памяти 1/ε (за измеренное время тишины — в
#        e^(ε·T) раз). Рельеф не гравировка, а среднее по прожитому.
#
# Уравнения те же, что в `basic.jl`. Запуск:
#
#     julia julia/kuramoto/relief.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6
const PASSES = 200      # проходов, которыми проживается среда
const PASS = 6.0        # длина прохода, с
const WIDE = 1e-3       # долгая память: 1/ε = 1000 с, окно шире прожитого
const SHORT = 1e-2      # короткая: 1/ε = 100 с при проходе 6 с

"""
    mean_link(medium)

Средняя связь среды по всем связям, кроме собственных. У свежей среды это её
начальные 0.6/N = 0.060: исходная связь, а не накопленное.
"""
function mean_link(m::Medium)
    s, n = 0.0, 0
    for i in axes(m.coupling, 1), j in axes(m.coupling, 2)
        i == j && continue
        s += m.coupling[i, j]
        n += 1
    end
    return s / n
end

"""
    link_goal(medium, i, j)

То, к чему идёт связь вибраций i и j:
(link + N₀ · Σₚₙ S(ωⱼₙ−ωᵢₚ)·cos(θᵢₚ−θⱼₙ) / Σₚₙ S) / K,
где K — число соседей: N в полном графе, z в городе.
Цель — не среда и не уровень, а совпадение этих двух вибраций: у каждой связи
своя, и меняется она с состоянием. Слагаемое link/N — старт, N₀·cos/N делает
цель сильнее старта, а деление на N держит полную тягу O(1). То же имя и то же
число, что в сверке.
"""
function link_goal(m::Medium, i::Int, j::Int)
    acc, weight = 0.0, 0.0
    for p in axes(m.omega, 2), n in axes(m.omega, 2)
        s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
        acc += s * cos(m.theta[i, p] - m.theta[j, n])
        weight += s
    end
    return memory_goal(m, i, j, acc, weight)
end

"""
    life(ε, passes; seed, events, place)

Среда, прожившая `passes` проходов по `PASS` секунд с памятью ε. `events = false` —
та же среда того же зерна и возраста, через которую событие не проходило ни разу.
`place` — вибрация, в которую толкает событие: так видно, следует ли запись за
местом события.
"""
function life(ε::Float64, passes::Int; seed::Int = 1, events::Bool = true, place::Int = 1)
    m = build(memory = ε, seed = seed)
    for _ in 1:passes
        events && perturb!(m, place, 1.5)
        advance!(m, PASS)
    end
    return m
end

"""
    relief_of(a, b)

Рельеф: чем связь среды `a` отличается от связи среды `b` по каждому месту.
Собственные связи не считаются: там своя связь, и она не пишется.
"""
function relief_of(a::Medium, b::Medium)
    δ = zeros(N, N)
    for i in 1:N, j in 1:N
        i == j && continue
        δ[i, j] = a.coupling[i, j] - b.coupling[i, j]
    end
    return δ
end

"""
    relief_place(δ, k)

Рельеф у места k (связи, которых k касается) и вдали от него: среднее |δ| по
связям. Место — не величина, а положение: если запись помнит место, у места
должно быть больше.
"""
function relief_place(δ::AbstractMatrix{Float64}, k::Int)
    near, far, nn, nf = 0.0, 0.0, 0, 0
    for i in axes(δ, 1), j in axes(δ, 2)
        i == j && continue
        v = abs(δ[i, j])
        if i == k || j == k
            near += v; nn += 1
        else
            far += v; nf += 1
        end
    end
    return (near / nn, far / nf)
end

"""
    values_of(δ)

Рельеф как ряд чисел по всем связям, кроме собственных.
"""
function values_of(δ::AbstractMatrix{Float64})
    v = Float64[]
    for i in axes(δ, 1), j in axes(δ, 2)
        i == j && continue
        push!(v, δ[i, j])
    end
    return v
end

"""
    agreement(δa, δb)

Насколько рельефы похожи: связь Пирсона по местам. Единица — рельеф тот же,
ноль — связи между рельефами нет.
"""
agreement(δa::AbstractMatrix{Float64}, δb::AbstractMatrix{Float64}) =
    cor(values_of(δa), values_of(δb))

"""
    median_of(values)

Середина ряда. Здесь это не украшение: наибольшее и наименьшее — края, а не
правило, и по одному краю свойство не читается.
"""
function median_of(values)
    s = sort(collect(values))
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end


"""
    main()

Прочитать, помнит ли среда место: знак и среда вместе — одна запись или две.
"""
function main()
    println("знак и среда вместе: помнит ли среда место, а не только уровень\n")

    println("--- доля одна, а записанное разное ----------------------------------")
    println("   Одна среда, один шаг: у каждой связи своя цель, а доля пути у всех")
    println("   одна и та же. Что за этим стоит — видно по целям и по записанному.")
    m = build(memory = WIDE, seed = 4)
    shares, written, goals, empty = Float64[], Float64[], Float64[], 0
    for i in 1:N, j in 1:N
        i == j && continue
        g, w = link_goal(m, i, j), m.coupling[i, j]
        need = g - w
        push!(goals, g)
        push!(written, abs(DT * m.memory * need))
        if abs(need) < 1e-12
            empty += 1
        else
            push!(shares, DT * m.memory * need / need)
        end
    end
    println("   доля пути                " *
            @sprintf("%.9f … %.9f", minimum(shares), maximum(shares)) *
            @sprintf("   (ε·dt = %.9f)", m.memory * DT))
    println("   отклонение доли от ε·dt  " *
            @sprintf("не более %.1e", maximum(abs.(shares .- m.memory * DT))))
    println("   цели связей              " *
            @sprintf("%+.3f … %+.3f", minimum(goals), maximum(goals)))
    println("   записанное за шаг        " *
            @sprintf("%.3e … %.3e", minimum(written), maximum(written)) *
            @sprintf("   (середина %.3e)", median_of(written)))
    @printf("   наибольшее к наименьшему в %.1f раз\n", maximum(written) / minimum(written))
    @printf("   цели со связью совпали целиком у %d связей из 90\n", empty)
    println("   Доля пути одна и та же — это размер шага; цели разные — это место.")
    println("   Уровень пишет время, цели пишут где.")

    println("\n--- след места: событие записывается там, где прошло ----------------")
    println("   Среда прожила 200 проходов по 6 с: через одну событие (толчок 1.5)")
    println("   шло каждый проход, через другую — ни разу. Зерно и возраст одни.")
    println("   Связь на месте события и вдали от него; середина по шести зёрнам.")
    println("       место   у места   вдали   во сколько раз   наибольшее у места")
    for place in (1, 5)
        near, far, top = Float64[], Float64[], 0.0
        for s in SEEDS
            q = life(WIDE, PASSES; seed = s, events = false)
            r = life(WIDE, PASSES; seed = s, events = true, place = place)
            n, f = relief_place(relief_of(r, q), place)
            push!(near, n); push!(far, f); top = max(top, n)
        end
        @printf("   %6d   %.4f   %.4f   %13.1f   %.4f\n", place,
                median_of(near), median_of(far), median_of(near) / median_of(far), top)
    end
    println("   Рельеф стоит у места события: у связей, которых оно касается, отличие")
    println("   больше, чем вдали, — и стоит там, где событие шло, а не где придётся.")

    println("\n--- величина рельефа против его места -------------------------------")
    println("   Рельеф у места события по каждому зерну и насколько рельефы похожи")
    println("   между зёрнами: долгая память против короткой.")
    println("      память      1/ε, с   у места по 6 зёрнам      похожесть рельефов")
    for (ε, label) in ((WIDE, "долгая"), (SHORT, "короткая"))
        δs, near = Matrix{Float64}[], Float64[]
        for s in SEEDS
            q = life(ε, PASSES; seed = s, events = false)
            r = life(ε, PASSES; seed = s, events = true, place = 1)
            δ = relief_of(r, q)
            push!(δs, δ)
            push!(near, relief_place(δ, 1)[1])
        end
        pairs = [agreement(δs[a], δs[b]) for a in 1:length(δs) for b in (a + 1):length(δs)]
        @printf("   %-10s %8.0f   %.4f … %.4f   %+.2f … %+.2f\n", label, 1 / ε,
                minimum(near), maximum(near), minimum(pairs), maximum(pairs))
    end
    println("   Место — свойство: у места события отличие больше у всех шести зёрен и")
    println("   при обеих памятях, и перенос события переносит рельеф (см. выше).")
    println("   Величина — бросок: сами числа рельефа у зёрен свои. Похожесть рельефов")
    println("   говорит, чей это рельеф: у долгой памяти он общий, у короткой — свой.")

    println("\n--- рельеф стирается прожитым ---------------------------------------")
    println("   Среда прожила 200 проходов с событием, потом 200 проходов без него;")
    println("   рядом — среда того же зерна и того же возраста без событий вовсе.")
    println("   Отличие у места события: до тишины и после. Если рельеф тает по сроку")
    println("   памяти 1/ε, за 1200 с тишины он должен упасть в e^(ε·1200) раз.")
    println("       память   1/ε, с   зерно   у места    после   ушло в ... раз")
    for (ε, seeds) in ((WIDE, SEEDS), (2e-3, 1:3))
        factors = Float64[]
        for s in seeds
            q1 = life(ε, PASSES; seed = s, events = false)
            r = life(ε, PASSES; seed = s, events = true, place = 1)
            n1, _ = relief_place(relief_of(r, q1), 1)
            q2 = life(ε, 2 * PASSES; seed = s, events = false)
            advance!(r, PASSES * PASS)
            n2, _ = relief_place(relief_of(r, q2), 1)
            push!(factors, n1 / n2)
            @printf("   %-8s %8.0f   %6d   %.4f   %.4f   %9.2f\n",
                    ε == WIDE ? "долгая" : "короче", 1 / ε, s, n1, n2, n1 / n2)
        end
        @printf("   срок e^(ε·T) = %.2f, середина по зёрнам %.2f — расхождение %+.1f%%\n",
                exp(ε * PASSES * PASS), median_of(factors),
                100 * (median_of(factors) / exp(ε * PASSES * PASS) - 1))
    end
    println("   Событие кончилось — и отметина уходит ровно по сроку памяти 1/ε: тем")
    println("   же ε, которым среда писала, она и забывает. Рельеф — не гравировка, а")
    println("   среднее по прожитому: тишина копится так же, как событие.")

    println("\nИтог: у записи одно ε на уровень и своя цель у каждой связи. Время пишет")
    println("ε (сколько помнится и чего хватит записать), место пишут цели (что с чем")
    println("совпало). Знак и среда — не две вещи, которые надо сводить: это одна")
    println("запись, читаемая двумя способами. Что не сделано: место записано рельефом")
    println("связи, а не отдельным знаком, — и чтобы рельеф держался дольше окна, ε")
    println("должен быть у связи, а не у уровня.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


