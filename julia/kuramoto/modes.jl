# Entrain. Взаимодействие мод.
#
# Это проверка, а не описание. Уравнения те же, что в basic.jl, а вопрос один:
# что даёт взаимодействие мод внутри вибрации.
#
# Выбор уже сделан, и он не в числах: мода — не отдельная сущность, а вибрация
# меньшего масштаба. Значит внутри вибрации должна работать та же среда, и
# спрашивать надо не «как связать моды», а «что из этого следует». Ниже —
# три чтения подряд: та же среда на меньшем масштабе, та же среда с узкой
# полосой, и отклик по совпадению отношения. Третье — единственное, чего нет
# в basic.jl: это не описание, а проба, и в ней меняется ровно одна вещь.
#
# Запуск:
#
#     julia julia/kuramoto/modes.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6

"""
    ratio_response(ratio, want, band)

Отклик по совпадению отношения: R(Δ, Δ₀) = 1 / (1 + ((Δ − Δ₀)/Πʀ)²).

То же, что overlap, но сравнивает не частоты, а их отношения. Обертоны
стоят далеко друг от друга по частоте и не слышат друг друга; отношения
у них стоят на одном месте, и по отношению они слышат.
"""
ratio_response(ratio::Float64, want::Float64, band::Float64) =
    1 / (1 + ((ratio - want) / band)^2)

"""
    ratio_norm(modes)

Отношения, которые у аккорда считаются своими: p/q для пары (p, q).
"""
ratio_norm(modes::Int) = [(p == q ? 1.0 : p / q) for p in 1:modes, q in 1:modes]

"""
    step_held!(medium, dt, strength, band, want, eps)

Шаг описания с одной заменой: внутренняя связь идёт по совпадению отношения.
Мода p тянется к отношению want[p, q] с модой q в той мере, в какой это
отношение уже почти такое. Всё остальное — как в step!.

При eps > 0 среда подтягивает и саму норму к тому, что через неё звучит:
не к тому, что положено, а к тому, что у неё уже почти получилось. При
eps = 0 и want = ratio_norm(modes) это в точности прежнее чтение.
"""
function step_held!(m::Medium, dt::Float64, strength::Float64, bandr::Float64,
                    want::Matrix{Float64}, eps::Float64 = 0.0,
                    weighted::Bool = true)
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
    for i in 1:vibs, p in 1:modes, q in 1:modes
        q == p && continue
        r = ratio_response(m.omega[i, p] / m.omega[i, q], want[p, q], bandr)
        drive[i, p] += strength * r * (want[p, q] * m.omega[i, q] - m.omega[i, p])
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
    if eps != 0.0
        # Норма меняется тем, что звучит. Взвешенно — только там, где это уже
        # почти норма; без веса — всюду, чем бы среда ни звучала.
        acc = zeros(modes, modes)
        weight = zeros(modes, modes)
        for i in 1:vibs, p in 1:modes, q in 1:modes
            q == p && continue
            ratio = m.omega[i, p] / m.omega[i, q]
            r = weighted ? ratio_response(ratio, want[p, q], bandr) : 1.0
            acc[p, q] += r * ratio
            weight[p, q] += r
        end
        for p in 1:modes, q in 1:modes
            q == p && continue
            weight[p, q] == 0.0 && continue
            want[p, q] += dt * eps * (acc[p, q] / weight[p, q] - want[p, q])
        end
    end
    return m
end

"""
    advance_held!(medium, seconds, strength, band, want, eps = 0.0)

Прожить средой указанное время по шагу с откликом по отношению.
"""
function advance_held!(m::Medium, seconds::Float64, strength::Float64,
                       bandr::Float64, want::Matrix{Float64}, eps::Float64 = 0.0,
                       weighted::Bool = true)
    for _ in 1:round(Int, seconds / DT)
        step_held!(m, DT, strength, bandr, want, eps, weighted)
    end
    return m
end

"""
    advance_ratio!(medium, seconds, strength, band)

То же, но норма не меняется: среда ничего не запоминает.
"""
advance_ratio!(m::Medium, seconds::Float64, strength::Float64, bandr::Float64) =
    advance_held!(m, seconds, strength, bandr, ratio_norm(size(m.omega, 2)), 0.0)

"Средний отклик между разными модами одной вибрации: слышат ли они друг друга."
function inner_overlap(m::Medium)
    modes = size(m.omega, 2)
    modes < 2 && return 0.0
    acc, n = 0.0, 0
    for i in 1:size(m.omega, 1), p in 1:modes, q in 1:modes
        p == q && continue
        acc += overlap(m.omega[i, q] - m.omega[i, p], m.band)
        n += 1
    end
    return acc / n
end


function main()
    @printf("взаимодействие мод: шесть начальных условий, наименьшее … наибольшее\n")

    println("\n--- то же, что между вибрациями, но внутри вибрации " * "-"^12)
    println("   modes = 3, полоса широкая: моды слышат друг друга.")
    for strength in (0.0, 0.6, 1.5, 4.0)
        wide = [(begin
            m = build(modes = 3, inner = strength, memory = 0.0, seed = s)
            advance!(m, 60.0)
            (unison(m, 1), tuning_error(m, 1), coherence(m))
        end) for s in SEEDS]
        @printf("   связь внутри %.1f   сжатие аккорда %.3f … %.3f   строй %.3f … %.3f   целое %.3f … %.3f\n",
                strength, minimum(w[1] for w in wide), maximum(w[1] for w in wide),
                minimum(w[2] for w in wide), maximum(w[2] for w in wide),
                minimum(w[3] for w in wide), maximum(w[3] for w in wide))
    end
    println("   Сжатие растёт до единицы, а ошибка отношения — вместе с ним: моды")
    println("   идут к одной частоте, а не к отношению. Та же среда внутри вибрации")
    println("   отношения не держит: она держит совпадение.")

    println("\n--- та же связь, но полоса узкая: слышат ли обертоны друг друга " * "-"^4)
    for band in (Inf, 0.3, 0.1)
        heard = [(begin
            m = build(modes = 3, band = band, inner = 1.5, memory = 0.0, seed = s)
            advance!(m, 60.0)
            (inner_overlap(m), unison(m, 1), tuning_error(m, 1))
        end) for s in SEEDS]
        @printf("   полоса %-4s   отклик внутри вибрации %.4f … %.4f   сжатие %.3f … %.3f   строй %.3f … %.3f\n",
                band == Inf ? "∞" : string(band),
                minimum(h[1] for h in heard), maximum(h[1] for h in heard),
                minimum(h[2] for h in heard), maximum(h[2] for h in heard),
                minimum(h[3] for h in heard), maximum(h[3] for h in heard))
    end
    println("   При узкой полосе обертоны друг друга не слышат вовсе: их частоты")
    println("   далеко, а отклик по совпадению частот на далёкое не отвечает.")

    println("\n--- отклик по совпадению отношения: то, чего нет в basic.jl " * "-"^4)
    println("   modes = 3, полоса широкая, λ = 0; внутренняя связь — по отношению.")
    for bandr in (0.05, 0.1, 0.3, 1.0)
        held = [(begin
            m = build(modes = 3, memory = 0.0, seed = s)
            advance_ratio!(m, 60.0, 8.0, bandr)
            (tuning_error(m, 1), unison(m, 1))
        end) for s in SEEDS]
        @printf("   полоса по отношению %.2f   строй %.4f … %.4f   сжатие %.3f … %.3f\n",
                bandr, minimum(h[1] for h in held), maximum(h[1] for h in held),
                minimum(h[2] for h in held), maximum(h[2] for h in held))
    end
    println("   Отношение держится — и при этом аккорд не сжимается: связь идёт")
    println("   не к одной частоте, а к отношению между частотами.")

    println("\n--- чем это отличается от λ " * "-"^28)
    println("   толчок обертону: сдвинуть вторую моду на 0.3 и прожить минуту.")
    for bandr in (0.05, 0.3, 1.0)
        after = [(begin
            m = build(modes = 3, memory = 0.0, seed = s)
            advance_ratio!(m, 60.0, 8.0, bandr)
            perturb!(m, 1, 0.3, 2)
            advance_ratio!(m, 60.0, 8.0, bandr)
            tuning_error(m, 1)
        end) for s in SEEDS]
        @printf("   отклик по отношению %.2f   строй после толчка %.4f … %.4f\n",
                bandr, minimum(after), maximum(after))
    end
    for strength in (2.0, 8.0)
        stiff = [(begin
            m = build(modes = 3, tuning = strength, memory = 0.0, seed = s)
            advance!(m, 60.0)
            perturb!(m, 1, 0.3, 2)
            advance!(m, 60.0)
            tuning_error(m, 1)
        end) for s in SEEDS]
        @printf("   строй λ = %.1f            строй после толчка %.4f … %.4f\n",
                strength, minimum(stiff), maximum(stiff))
    end
    println("   Толчок обертону возвращают оба: и λ, и отклик по отношению. Разница")
    println("   в другом — и видна она на толчке основной.")
    println("\n   толчок основной: сдвинуть первую моду на 0.3, затухание выключено,")
    println("   чтобы возвращала именно связь. Смотрим и частоту, и отношение.")
    for bandr in (0.1, 0.3)
        back = [(begin
            m = build(modes = 3, damping = 0.0, memory = 0.0, seed = s)
            advance_ratio!(m, 60.0, 8.0, bandr)
            perturb!(m, 1, 0.3, 1)
            advance_ratio!(m, 60.0, 8.0, bandr)
            (drift(m, 1), tuning_error(m, 1))
        end) for s in SEEDS]
        @printf("   отклик по отношению %.2f   отклонение основной %.4f … %.4f   строй %.4f … %.4f\n",
                bandr, minimum(b[1] for b in back), maximum(b[1] for b in back),
                minimum(b[2] for b in back), maximum(b[2] for b in back))
    end
    for strength in (2.0, 8.0)
        back = [(begin
            m = build(modes = 3, damping = 0.0, tuning = strength, memory = 0.0, seed = s)
            advance!(m, 60.0)
            perturb!(m, 1, 0.3, 1)
            advance!(m, 60.0)
            (drift(m, 1), tuning_error(m, 1))
        end) for s in SEEDS]
        @printf("   строй λ = %.1f            отклонение основной %.4f … %.4f   строй %.4f … %.4f\n",
                strength, minimum(b[1] for b in back), maximum(b[1] for b in back),
                minimum(b[2] for b in back), maximum(b[2] for b in back))
    end
    println("   λ держит отношение и не держит основную: ни одно начальное условие")
    println("   её не вернуло, и отклонение осталось в разы больше самого толчка.")
    println("   Отклик по отношению держит и её: у части начальных условий основная")
    println("   возвращается почти совсем, у части — уезжает. Гарантии нет, но")
    println("   привязка есть, и отношение он держит строже, чем λ.")

    println("\n--- след внутри вибрации " * "-"^32)
    for inner_memory in (0.0, 1e-3)
        trace = [(begin
            m = build(modes = 3, inner = 1.5, inner_memory = inner_memory,
                      memory = 0.0, seed = s)
            was = inner_link(m)
            advance!(m, 60.0)
            (was, inner_link(m), tuning_error(m, 1))
        end) for s in SEEDS]
        @printf("   память внутри %.0e   связь внутри %.2f → %.4f … %.4f   строй %.3f … %.3f\n",
                inner_memory, first(trace)[1],
                minimum(t[2] for t in trace), maximum(t[2] for t in trace),
                minimum(t[3] for t in trace), maximum(t[3] for t in trace))
    end
    println("   Внутренняя связь наводится тем, что через неё прошло, — так же,")
    println("   как связь между вибрациями. И так же, как там, отношения она не")
    println("   создаёт: она закрепляет то, что уже почти было.")

    println("\n--- строй как след: запоминает ли среда само отношение " * "-"^2)
    println("   толчок обертону на 0.5, затем минута; λ = 0, γ = 0.8, отклик по")
    println("   отношению с полосой 0.3. Норма — то, что среда считает отношением.")
    for (strength, eps) in ((8.0, 0.0), (8.0, 1e-1), (0.5, 0.0), (0.5, 1e-2), (0.5, 1e-1))
        seen = [(begin
            m = build(modes = 3, memory = 0.0, seed = s)
            want = ratio_norm(3)
            advance_held!(m, 60.0, strength, 0.3, want, eps)
            perturb!(m, 1, 0.5, 2)
            advance_held!(m, 60.0, strength, 0.3, want, eps)
            (tuning_error(m, 1), want[2, 1], want[3, 1])
        end) for s in SEEDS]
        @printf("   связь %.1f, память %.0e   строй после толчка %.4f … %.4f   норма %.4f … %.4f и %.4f … %.4f\n",
                strength, eps,
                minimum(x[1] for x in seen), maximum(x[1] for x in seen),
                minimum(x[2] for x in seen), maximum(x[2] for x in seen),
                minimum(x[3] for x in seen), maximum(x[3] for x in seen))
    end
    println("   При сильной связи норма почти не двигается: толчок возвращается")
    println("   быстрее, чем успевает записаться, и прошлое нигде не остаётся.")
    println("   При слабой — толчок остаётся в состоянии, и тогда норма едет за")
    println("   ним: среда запоминает не толчок, а то, чем стала. След получается")
    println("   не от события, а от того, что событие оставило вибрацию другой.")
    println("   У русла между вибрациями всё иначе: там прошлое остаётся в самой")
    println("   связи, потому что согласованность фаз накапливается во времени.")

    return nothing
end

main()