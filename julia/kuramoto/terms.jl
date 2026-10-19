# Entrain. Курамото: срок памяти — у связи или у уровня?
#
# В главе 22 стоял шаг записи, названный дважды и с двух сторон: в масштабах
# памяти — чтобы «мгновенное» и «годовое» стояли рядом в одной среде, — и при
# чтении места — чтобы написанное пережило окно. Шаг один: ε не у уровня, а у
# каждой связи.
#
# Здесь он делается и читается тем же, чем читалась память:
#
#     1. это не вторая запись: если ε у всех связей одно, числа среды те же до
#        последнего бита. Раскладка ε по связям — тот же закон, а не новый;
#     2. доля пути у каждой связи своя — ровно её ε·dt; а записанное за шаг — не
#        одно число на уровень: путь к цели у каждой связи свой;
#     3. рядом в одной среде: быстрая связь к концу прожитого стоит у своей
#        цели, медленная — при своей. «Мгновенное» и «годовое» описываются
#        вместе — тем, чего не хватало одному ε на уровень;
#     4. написанное живёт по своему сроку: рельеф места у медленной связи
#        переживает окно, у быстрой — уходит вместе с ним (за T секунд тишины
#        в e^(ε·T) раз, у каждой по своему ε).
#
# Уравнения те же, что в `basic.jl`. Новое место одно и названо в записи:
# `memory_matrix` — ε у каждой связи; когда его нет, у уровня одно ε, и запись
# прежняя. Уровень тут не только среда: у мира внутри вибрации тот же закон и
# та же раскладка (`inner_memory_matrix`).
#
# Запуск:
#
#     julia julia/kuramoto/terms.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6
const PASSES = 200      # проходов, которыми проживается среда
const PASS = 6.0        # длина прохода, с
const LEVEL = 1e-3      # ε уровня: 1/ε = 1000 с, окно шире прожитого (1200 с)
const FAST = 1e-2       # ε связи на месте: 1/ε = 100 с — «мгновенное»
const SLOW = 1e-4       # ε другой связи того же места: 1/ε = 10000 с — «годовое»

"""
    terms(; level, fast, slow)

Расклад ε по связям: у всех — ε уровня, у связи (1,2) — быстрая, у связи (1,3) —
медленная. Обе связи — у одного места (обе трогают вибрацию 1): так видно, что
срок — не место, а своё число у каждой связи.
"""
function terms(; level::Float64 = LEVEL, fast::Float64 = FAST, slow::Float64 = SLOW)
    ε = fill(level, N, N)
    ε[1, 2] = fast
    ε[1, 3] = slow
    return ε
end

"""
    life(written, passes; seed, place)

Среда, прожившая `passes` проходов по `PASS` секунд. `written` — либо ε уровня
(одно число), либо расклад ε по связям. `place = 0` — та же среда того же зерна
и возраста, через которую событие не проходило ни разу.
"""
function life(written, passes::Int; seed::Int = 1, place::Int = 0)
    m = written isa Float64 ? build(memory = written, seed = seed) :
                              build(memory_matrix = written, seed = seed)
    for _ in 1:passes
        place != 0 && perturb!(m, place, 1.5)
        advance!(m, PASS)
    end
    return m
end

"""
    link_goal(medium, i, j)

То, к чему идёт связь вибраций i и j:
link · Σₚₙ S(ωⱼₙ−ωᵢₚ)·cos(θᵢₚ−θⱼₙ) / (N·Σₚₙ S).
То же имя и то же число, что в сверке и в сцене места: цель — не среда и не
уровень, а совпадение этих двух вибраций. Деление на N держит связь на пару
O(1/N): иначе полная тяга росла бы с числом вибраций, а множитель link
возвращает памяти масштаб среды.
"""
function link_goal(m::Medium, i::Int, j::Int)
    acc, weight = 0.0, 0.0
    for p in axes(m.omega, 2), n in axes(m.omega, 2)
        s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
        acc += s * cos(m.theta[i, p] - m.theta[j, n])
        weight += s
    end
    return weight == 0.0 ? m.coupling[i, j] :
           m.link * acc / (weight * size(m.coupling, 1))
end
function main()
    println("срок памяти: у связи или у уровня?\n")

    println("--- это не вторая запись: ε у всех связей одно ---------------------")
    @printf("   Среда с ε уровня и среда с тем же ε, разложенным по связям, прожиты\n")
    @printf("   %d с; сравниваются все числа — состояние и связь.\n", PASSES * PASS)
    println("        ε   |разница частот|   |разница фаз|   |разница связи|")
    for ε in (1e-5, 1e-3, 1e-2)
        a = build(memory = ε, seed = 4)
        b = build(memory = ε, memory_matrix = fill(ε, N, N), seed = 4)
        advance!(a, 1200.0)
        advance!(b, 1200.0)
        @printf("   %6.0e   %.1e           %.1e         %.1e\n", ε,
                maximum(abs.(a.omega .- b.omega)),
                maximum(abs.(a.theta .- b.theta)),
                maximum(abs.(a.coupling .- b.coupling)))
    end
    println("   То же у мира внутри вибрации: εᵤ у всех пар мод одно — числа те же.")
    a = build(modes = 3, band = 0.1, tuning = 8.0, inner = 0.5, inner_memory = 1e-4,
              seed = 4)
    b = build(modes = 3, band = 0.1, tuning = 8.0, inner = 0.5, inner_memory = 1e-4,
              inner_memory_matrix = fill(1e-4, 3, 3), seed = 4)
    advance!(a, 1200.0)
    advance!(b, 1200.0)
    @printf("   разница по мирам: частоты %.1e, связи мод %.1e\n",
            maximum(abs.(a.omega .- b.omega)),
            maximum([maximum(abs.(world(a, i).coupling .- world(b, i).coupling))
                     for i in 1:N]))
    println("   Ноль, а не «почти»: раскладка ε — та же запись, разложенная по связям,")
    println("   и при одном ε у всех она не добавляет ни одного нового числа.")

    println("\n--- доля пути у каждой связи своя -----------------------------------")
    println("   Один шаг среды с раскладкой ε: записанное за шаг против своего ε·dt.")
    println("        связь   ε          ε·dt         доля за шаг   откл.        записано")
    m = build(memory_matrix = terms(), seed = 4)
    was = copy(m.coupling)
    step!(m, DT)
    written = Float64[]
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        goal = link_goal(m, i, j)
        target = goal - was[i, j]
        share = (m.coupling[i, j] - was[i, j]) / target
        push!(written, m.coupling[i, j] - was[i, j])
        @printf("   (%2d,%2d)   %.0e   %.3e   %.9f   %.1e   %+.6f\n", i, j,
                m.memory_matrix[i, j], m.memory_matrix[i, j] * DT, share,
                abs(share - m.memory_matrix[i, j] * DT), m.coupling[i, j] - was[i, j])
    end
    @printf("   Доля — не одно число на уровень, а у каждой связи своё: у быстрой связи\n")
    @printf("   она в %.0f раз больше, чем у медленной, — а записано за шаг в %.0f раз\n",
            m.memory_matrix[1, 2] / m.memory_matrix[1, 3], abs(written[1]) / abs(written[2]))
    println("   больше, и это не то же число: путь к цели у этих связей разной длины.")

    println("\n--- мгновенное рядом с годовым в одной среде ------------------------")
    fresh = build(memory_matrix = terms(), seed = SEEDS[1]).coupling[1, 3]
    @printf("   1200 с жизни, толчок в первую вибрацию каждый из %d проходов. Быстрая\n", PASSES)
    @printf("   связь (1,2): 1/ε = %.0f с, ε·прожитое = %.2f. Медленная (1,3): 1/ε = %.0f с,\n",
            1 / terms()[1, 2], terms()[1, 2] * PASSES * PASS,
            1 / terms()[1, 3])
    @printf("   ε·прожитое = %.2f. Свежая связь — %.3f.\n",
            terms()[1, 3] * PASSES * PASS, fresh)
    println("        зерно   связь (1,2)   связь (1,3)   медленная от своей")
    fast_link, slow_link = Float64[], Float64[]
    for s in SEEDS
        m = life(terms(), PASSES; seed = s, place = 1)
        push!(fast_link, m.coupling[1, 2])
        push!(slow_link, m.coupling[1, 3])
        @printf("   %9d   %+.4f       %+.4f       %+.4f\n", s, m.coupling[1, 2],
                m.coupling[1, 3], m.coupling[1, 3] - fresh)
    end
    @printf("   Быстрая связь прошла свой путь целиком (%.3f…%.3f против свежих %.3f), а\n",
            minimum(fast_link), maximum(fast_link), fresh)
    @printf("   медленная — только часть: ε·прожитое у неё %.2f, то есть около восьмой\n",
            terms()[1, 3] * PASSES * PASS)
    @printf("   части пути, — отсюда и связь %.3f…%.3f против свежих %.3f. В одной среде\n",
            minimum(slow_link), maximum(slow_link), fresh)
    println("   стоят рядом оба срока и не спорят — у каждой связи своё ε.")

    println("\n--- написанное живёт по своему сроку --------------------------------")
    @printf("   Место (первая вибрация) помнит две связи с разным сроком. После %.1f с\n",
            PASSES * PASS)
    println("   тишины сравнивается среда того же зерна и возраста без событий: то,")
    println("   чем связи отличаются, и есть написанное. Осталось — доля от записанного")
    @printf("   за жизнь; ждётся e^(-ε·T): у быстрой %.1e, у медленной %.3f.\n",
            exp(-terms()[1, 2] * PASSES * PASS), exp(-terms()[1, 3] * PASSES * PASS))
    println("        зерно   |δ(1,2)|: жизнь → тишина   осталось   |δ(1,3)|: жизнь → тишина   осталось")
    fast_kept, slow_kept = Float64[], Float64[]
    for s in SEEDS
        ε = terms()
        routed = life(ε, PASSES; seed = s, place = 1)
        quiet = life(ε, PASSES; seed = s)
        f1 = abs(routed.coupling[1, 2] - quiet.coupling[1, 2])
        s1 = abs(routed.coupling[1, 3] - quiet.coupling[1, 3])
        advance!(routed, PASSES * PASS)
        aged = life(ε, 2 * PASSES; seed = s)
        f2 = abs(routed.coupling[1, 2] - aged.coupling[1, 2])
        s2 = abs(routed.coupling[1, 3] - aged.coupling[1, 3])
        push!(fast_kept, f2 / f1)
        push!(slow_kept, s2 / s1)
        @printf("   %9d   %.5f → %.8f   %.2e   %.5f → %.5f   %.3f\n", s, f1, f2,
                f2 / f1, s1, s2, s2 / s1)
    end
    @printf("   по шести зёрнам: у быстрой осталось %.1e … %.1e, у медленной %.3f … %.3f\n",
            minimum(fast_kept), maximum(fast_kept), minimum(slow_kept), maximum(slow_kept))
    @printf("   Обе связи — у одного места, и рельеф у обеих написан одним событием.\n")
    println("   Стирает его тишина по-разному: у быстрой от написанного почти ничего не")
    @printf("   осталось (%.1e … %.1e), а у медленной — %.3f…%.3f, как и ждалось.\n",
            minimum(fast_kept), maximum(fast_kept), minimum(slow_kept), maximum(slow_kept))

    println("\nИтог: ε у связи — не второе число и не новая математика, а то же ε, у каждой")
    println("связи своё. При одном ε у всех запись прежняя до последнего бита; при разных")
    println("в одной среде стоят рядом мгновенное и годовое, и написанное переживает окно")
    println("там, где срок длиннее. Что отсюда следует для места: рельеф связи держится")
    println("не дольше своего срока, поэтому долгим его делает медленная связь, а не")
    println("место, где шло событие. Само написанное (откуда берётся промах r₀) этим не")
    println("тронуто и остаётся открытым в главе 22.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
