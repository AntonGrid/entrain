# Entrain. Взаимодействие мод.
#
# Это проверка, а не описание. Уравнения те же, что в basic.jl, а вопрос один:
# что даёт взаимодействие мод внутри вибрации.
#
# Выбор уже сделан, и он не в числах: мода — не отдельная сущность, а вибрация
# меньшего масштаба. Значит внутри вибрации должна работать та же среда, и
# спрашивать надо не «как связать моды», а «что из этого следует». Ниже —
# чтения подряд: та же среда на меньшем масштабе, та же среда с узкой полосой,
# отклик по совпадению отношения, связь по кратной фазе и, наконец, уровни
# моды — шкала обертонов, которой можно слышать чужую моду. Последнее — не
# описание, а проба, и в нём меняется ровно одна вещь: задано, что шкала есть,
# и не задано никакого отношения. Вопрос там один: соберётся ли строй сам.
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

"""
    step_scale!(medium, dt, strength, link, eps)

Связь мод по кратной фазе. Сравниваются не частоты и не отношения частот,
а фаза, приведённая к масштабу моды: мода p отвечает на разность
θ_q − (q/p)·θ_p. Эта разность не дрейфует ровно тогда, когда отношение
ω_q/ω_p уже равно q/p: при других частотах она бежит, и отклик усредняется
в ноль сам собой. Полосы здесь нет — она не нужна.

`link[p, q]` — наведённая связь: при eps > 0 она тянется к усреднённому по
времени cos той же разности, то есть к тому, держалось ли это отношение.
"""
function step_scale!(m::Medium, dt::Float64, strength::Float64,
                     link::Matrix{Float64}, eps::Float64 = 0.0)
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
        drive[i, p] += strength * link[p, q] *
                       sin(m.theta[i, q] - (q / p) * m.theta[i, p])
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
        for p in 1:modes, q in 1:modes
            q == p && continue
            acc = 0.0
            for i in 1:vibs
                acc += cos(m.theta[i, q] - (q / p) * m.theta[i, p])
            end
            link[p, q] += dt * eps * (acc / vibs - link[p, q])
        end
    end
    return m
end

"""
    advance_scale!(medium, seconds, strength, link, eps = 0.0)

Прожить средой указанное время по шагу со связью по кратной фазе.
"""
function advance_scale!(m::Medium, seconds::Float64, strength::Float64,
                        link::Matrix{Float64}, eps::Float64 = 0.0)
    for _ in 1:round(Int, seconds / DT)
        step_scale!(m, DT, strength, link, eps)
    end
    return m
end

"Средняя наведённая связь по кратным парам (без диагонали)."
function scale_link(link::Matrix{Float64})
    modes = size(link, 1)
    modes < 2 && return 0.0
    acc = 0.0
    for p in 1:modes, q in 1:modes
        p == q && continue
        acc += link[p, q]
    end
    return acc / (modes * (modes - 1))
end


# --- возникает ли строй сам ------------------------------------------------

"""
    level_weights(K)

Уровни моды: k = 1…K — её собственные обертоны, её собственная шкала. Мода
отвечает уровню тем сильнее, чем ниже уровень: вес 1/(k·j).

Здесь не задано никакого отношения. Задано только то, что уровни есть, — а
это и есть «мода есть вибрация меньшего масштаба»: у вибрации есть моды, у
моды есть уровни.
"""
level_weights(K::Int) = [1.0 / (k * j) for k in 1:K, j in 1:K]

"""
    scatter!(medium, low, high, seed = 7)

Разбросать свои частоты мод по всему диапазону: каждая мода получает свою
частоту, ни к какой другой не кратную. Аккорда в среде не строят — отношения
в ней случайны. Так у строя нет возможности «уже быть»: его можно только
навести.
"""
function scatter!(m::Medium, low::Float64, high::Float64, seed::Int = 7)
    rng = MersenneTwister(seed)
    for i in 1:size(m.omega, 1), p in 1:size(m.omega, 2)
        m.omega0[i, p] = low + (high - low) * rand(rng)
    end
    m.omega .= m.omega0
    return m
end

"""
    level_error(medium, i, p, q, K)

Насколько хорошо отношение двух мод ложится на уровень: наименьшее из
|k·ω_p − j·ω_q| / (k·ω_p) по всем k, j ≤ K. Ноль — отношение выполняется
ровно, и пара (k, j) говорит, какое именно: ω_p / ω_q = j / k.
"""
function level_error(m::Medium, i::Int, p::Int, q::Int, K::Int)
    best, bi, bj = Inf, 1, 1
    for k in 1:K, j in 1:K
        e = abs(k * m.omega[i, p] - j * m.omega[i, q]) / (k * m.omega[i, p])
        if e < best
            best, bi, bj = e, k, j
        end
    end
    return best, bi, bj
end

"""
    step_levels!(medium, dt, strength, C, link, eps)

Шаг описания, в котором мода слышит другую своими уровнями: уровень k моды p
и уровень j моды q.

Отбор делает ближайший уровень. Мода пробует все свои уровни против всех
чужих, и работает тот, до которого близко: отклик S(k·ω_p − j·ω_q, Π) для
далёкого мал. Пока уровень близок, он тянет частоту моды к тому, что этот
уровень означает: ω_p → (j / k)·ω_q. Полосы по отношению здесь нет: Π — это
отклик уровня, и она же решает, сколько отношений вообще доступно.

Никакого отношения не задано — заданы только уровни. Какое отношение
получится, решает среда.

`link[k, j]` — наведённый вес уровня: при eps > 0 он тянется к усреднённому
cos той же разности, то есть к тому, держалось ли это отношение.
"""
function step_levels!(m::Medium, dt::Float64, strength::Float64,
                      C::Matrix{Float64}, link::Matrix{Float64},
                      eps::Float64 = 0.0)
    vibs, modes = size(m.omega)
    K = size(C, 1)
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
        for k in 1:K, j in 1:K
            weight = strength * C[k, j] * link[k, j]
            weight == 0.0 && continue
            near = overlap(k * m.omega[i, p] - j * m.omega[i, q], m.band)
            drive[i, p] += weight * near * (j * m.omega[i, q] / k - m.omega[i, p])
        end
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
        for k in 1:K, j in 1:K
            acc, n = 0.0, 0
            for i in 1:vibs, p in 1:modes, q in 1:modes
                q == p && continue
                acc += cos(k * m.theta[i, p] - j * m.theta[i, q])
                n += 1
            end
            n == 0 && continue
            link[k, j] += dt * eps * (acc / n - link[k, j])
        end
    end
    return m
end

"""
    advance_levels!(medium, seconds, strength, C, link, eps = 0.0)

Прожить средой указанное время по шагу со связью через уровни мод.
"""
function advance_levels!(m::Medium, seconds::Float64, strength::Float64,
                         C::Matrix{Float64}, link::Matrix{Float64},
                         eps::Float64 = 0.0)
    for _ in 1:round(Int, seconds / DT)
        step_levels!(m, DT, strength, C, link, eps)
    end
    return m
end

"""
    found_ratios(medium, K, tol)

Что среда нашла сама: по каждой паре мод — уровень (k, j), выполненный с
точностью лучше tol. Уровень (1, 1) означает, что две моды просто слились:
это не аккорд, а совпадение.
"""
function found_ratios(m::Medium, K::Int, tol::Float64)
    counts = Dict{Tuple{Int, Int}, Int}()
    pairs, close = 0, 0
    for i in 1:size(m.omega, 1), p in 1:size(m.omega, 2), q in (p + 1):size(m.omega, 2)
        pairs += 1
        e, bi, bj = level_error(m, i, p, q, K)
        if e < tol
            close += 1
            counts[(bi, bj)] = get(counts, (bi, bj), 0) + 1
        end
    end
    return close, pairs, counts
end

"Счёт уровней в строку: пара (k, j) значит ω_p / ω_q = j / k."
function found_line(counts::Dict{Tuple{Int, Int}, Int})
    isempty(counts) && return "ничего"
    items = sort(collect(counts), by = x -> (-x[2], x[1]))
    return join([@sprintf("%d:%d ×%d", j, k, n) for ((k, j), n) in items], "  ")
end

"""
    found_inside(medium, K, tol)

То же, но только пары мод одной вибрации: что среда сделала с аккордом, не
глядя на другие вибрации.
"""
function found_inside(m::Medium, K::Int, tol::Float64)
    counts = Dict{Tuple{Int, Int}, Int}()
    pairs, close = 0, 0
    for i in 1:size(m.omega, 1), p in 1:size(m.omega, 2), q in (p + 1):size(m.omega, 2)
        pairs += 1
        e, bi, bj = level_error(m, i, p, q, K)
        if e < tol
            close += 1
            counts[(bi, bj)] = get(counts, (bi, bj), 0) + 1
        end
    end
    return close, pairs, counts
end


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

"Наибольшие наведённые уровни: где осел след. (k, j) — это уровни, не отношение."
function link_top(link::Matrix{Float64}, n::Int)
    K = size(link, 1)
    items = [(link[k, j], k, j) for k in 1:K for j in 1:K]
    sort!(items, by = x -> -abs(x[1]))
    return join([@sprintf("(%d,%d) %.2f", k, j, v) for (v, k, j) in items[1:n]], "  ")
end

"Насколько сломан аккорд у одной вибрации: худшая пара, в долях."
function chord_error(m::Medium, i::Int)
    modes = size(m.omega, 2)
    worst = 0.0
    for p in 1:modes, q in (p + 1):modes
        worst = max(worst, abs(m.omega[i, p] / m.omega[i, q] * q / p - 1))
    end
    return worst
end

"Расстроить аккорды: обертоны сбиты со своих мест."
function detune!(m::Medium, spread::Float64, seed::Int)
    rng = MersenneTwister(seed)
    for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
        m.omega0[i, p] *= 1 + spread * randn(rng)
    end
    m.omega .= m.omega0
    return m
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

    println("\n--- связь по кратной фазе: та, которой полоса не нужна " * "-"^3)
    println("   мода p отвечает на θ_q − (q/p)·θ_p: эта разность бежит, если")
    println("   отношение не выполняется, и стоит, если выполняется. λ = 0.")
    for strength in (0.0, 2.0, 8.0)
        held = [(begin
            m = build(modes = 3, memory = 0.0, seed = s)
            link = ones(3, 3)
            advance_scale!(m, 60.0, strength, link)
            (tuning_error(m, 1), unison(m, 1))
        end) for s in SEEDS]
        @printf("   связь %.1f   строй %.4f … %.4f   сжатие %.3f … %.3f\n",
                strength, minimum(h[1] for h in held), maximum(h[1] for h in held),
                minimum(h[2] for h in held), maximum(h[2] for h in held))
    end
    println("\n   толчок обертону 0.5, затем минута: что осталось в связи.")
    for (strength, eps, link0) in ((8.0, 0.0, 1.0), (8.0, 1e-2, 1.0), (8.0, 1e-2, 0.1),
                                   (2.0, 1e-2, 1.0), (2.0, 1e-1, 1.0))
        seen = [(begin
            m = build(modes = 3, memory = 0.0, seed = s)
            link = fill(link0, 3, 3)
            advance_scale!(m, 60.0, strength, link, eps)
            perturb!(m, 1, 0.5, 2)
            advance_scale!(m, 60.0, strength, link, eps)
            (tuning_error(m, 1), scale_link(link), minimum(link))
        end) for s in SEEDS]
        @printf("   связь %.1f, память %.0e, начало %.1f   строй %.4f … %.4f   связь %.3f … %.3f   наименьшая %.3f … %.3f\n",
                strength, eps, link0,
                minimum(x[1] for x in seen), maximum(x[1] for x in seen),
                minimum(x[2] for x in seen), maximum(x[2] for x in seen),
                minimum(x[3] for x in seen), maximum(x[3] for x in seen))
    end
    println("   Без полосы, без λ и без выделенной моды: верхняя граница ошибки")
    println("   падает в разы (3–6), но не так ровно, как у λ — на части начальных")
    println("   условий аккорд не сжимается, а раздувается. Зато эта связь")
    println("   единственная, которая наводится: с памятью она растёт к тому, что")
    println("   держалось (начатая с 0.1 — в разы), а при сильной памяти уходит и в")
    println("   минус: среда запоминает не только строй, но и его отсутствие.")

    println("\n--- строй сам: заданы только уровни, отношения не задано " * "-"^4)
    println("   у каждой моды свои обертоны 1…K — её шкала. Мода слышит другую")
    println("   через ближайший свой уровень и тянет частоту к тому, что этот")
    println("   уровень означает. Отношения здесь нет: задано только то, что шкала")
    println("   есть. Аккорды сбиты со своих мест на 8%.")
    K = 4
    C = level_weights(K)
    detuned = [(begin
        m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
        detune!(m, 0.08, 900 + s)
        close, pairs, _ = found_inside(m, K, 0.02)
        (close, pairs, minimum(chord_error(m, i) for i in 1:N),
         maximum(chord_error(m, i) for i in 1:N))
    end) for s in SEEDS]
    @printf("   сбито: нашлось %2d из %2d   худшая пара %.4f … %.4f\n",
            sum(d[1] for d in detuned), sum(d[2] for d in detuned),
            minimum(d[3] for d in detuned), maximum(d[4] for d in detuned))
    for strength in (1.0, 4.0, 8.0, 16.0)
        res = [(begin
            m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
            detune!(m, 0.08, 900 + s)
            link = ones(K, K)
            advance_levels!(m, 60.0, strength, C, link)
            close, pairs, counts = found_inside(m, K, 0.02)
            (close, pairs, counts, mean(unison(m, i) for i in 1:N))
        end) for s in SEEDS]
        total = Dict{Tuple{Int, Int}, Int}()
        for r in res, (key, n) in r[3]
            total[key] = get(total, key, 0) + n
        end
        @printf("   связь %4.1f   нашлось %2d из %2d   сжатие %.3f … %.3f   %s\n",
                strength, sum(r[1] for r in res), sum(r[2] for r in res),
                minimum(r[4] for r in res), maximum(r[4] for r in res),
                found_line(total))
    end
    println("   Первая строка — без связи: сбитые аккорды остаются сбитыми, и")
    println("   найденное там — случайные совпадения (уровней много, попасть легко).")
    println("   Дальше видно: отношения приходят к целым — 2:1, 3:1, 3:2, — тем")
    println("   самым, которых никто не прописывал, и приходят не в слияние:")
    println("   сжатие держится около нуля.")

    println("\n   нужна ли для этого шкала: та же связь, разное число уровней.")
    for kk in (1, 2, 3, 4)
        res = [(begin
            m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
            detune!(m, 0.08, 900 + s)
            link = ones(kk, kk)
            advance_levels!(m, 60.0, 8.0, level_weights(kk), link)
            found_inside(m, kk, 0.02)[1]
        end) for s in SEEDS]
        @printf("   уровней %d   нашлось %2d из %2d\n", kk, sum(res), 3 * N * length(SEEDS))
    end
    println("   С одним уровнем связи хватает только на совпадение: аккорд не")
    println("   собирается. С трёх — собирается. Строю нужна шкала, а не связь")
    println("   вообще: целое берётся не из силы, а из того, что есть чем отвечать.")

    println("\n   толчок обертону 0.3, затем минута: возвращается ли собранное.")
    for (strength, eps) in ((16.0, 0.0), (16.0, 1e-2), (4.0, 1e-2))
        res = [(begin
            m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
            detune!(m, 0.08, 900 + s)
            link = ones(K, K)
            advance_levels!(m, 60.0, strength, C, link, eps)
            was = found_inside(m, K, 0.02)[1]
            perturb!(m, 1, 0.3, 2)
            advance_levels!(m, 60.0, strength, C, link, eps)
            (was, found_inside(m, K, 0.02)[1], link_top(link, 3))
        end) for s in SEEDS]
        @printf("   связь %4.1f, память %.0e   нашлось до толчка %2d … %2d, после %2d … %2d\n",
                strength, eps, minimum(r[1] for r in res), maximum(r[1] for r in res),
                minimum(r[2] for r in res), maximum(r[2] for r in res))
        @printf("%40s след: %s\n", "",
                eps == 0.0 ? "не наводится: память выключена" : res[1][3])
    end
    println("   Без памяти собранное возвращается ровно, а с памятью — хуже: в")
    println("   наведённый вес входит не только то, что стояло, но и то, что нет,")
    println("   поэтому он слабее заданного. Садится след на те же уровни, которыми")
    println("   аккорд и держался: (2,4) — это то же 1:2, взятое вдвое. Среда")
    println("   закрепляет не событие, а то, что сходилось.")

    println("\n   а если частоты не кратны ни к чему: 0.6 … 1.8, аккорда не строят.")
    for strength in (1.0, 8.0)
        res = [(begin
            m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
            scatter!(m, 0.6, 1.8, 700 + s)
            link = ones(K, K)
            advance_levels!(m, 60.0, strength, C, link)
            close, pairs, counts = found_ratios(m, K, 0.02)
            (close, pairs, counts)
        end) for s in SEEDS]
        total = Dict{Tuple{Int, Int}, Int}()
        for r in res, (key, n) in r[3]
            total[key] = get(total, key, 0) + n
        end
        @printf("   связь %4.1f   нашлось %3d из %3d   %s\n",
                strength, sum(r[1] for r in res), sum(r[2] for r in res),
                found_line(total))
    end
    println("   Среда находит и здесь — но в основном совпадение: отношение 1:1")
    println("   доступно всегда, а до 3:2 надо ещё дожить. Строй возникает из того,")
    println("   что шкала допускает и что оказалось близко.")

    return nothing
end

main()