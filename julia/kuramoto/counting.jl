# Entrain. Курамото: счёт — обертоны не написаны, а сосчитаны.
#
# Это проверка гипотезы (журнал, 2026-10-14, восьмой раз): у моды внутри —
# счёт себя. Обертон моды — не отдельная частота, которую где-то хранят, а
# сама мода, сосчитанная целое число раз: k-й уровень есть ω, взятая k раз.
# Уровня как величины нет — есть чтение. Значит обертоны можно не писать.
#
# Здесь это и сделано: у вибрации одна своя частота, а её уровни k·ω берутся
# из того же числа. Обертонов в состоянии нет вовсе — ни одного.
#
# Уравнения те же, что в basic.jl: связь, отклик S(Δ), затухание. Новое место
# одно и названо: уровни берутся у разных вибраций. У моды своей частоты нет,
# а уровни есть, поэтому счёт может работать только между вибрациями —
# k·ωᵢ ≈ j·ωⱼ. Это тот же шаг, что «строй сам» в modes.jl, только там он ходит
# по модам одной вибрации, а здесь — по вибрациям.
#
# Что спрашивается: держится ли при этом аккорд при λ = 0 — или, как в прошлых
# чтениях, без силы не держится ничего.
#
# Итог этого чтения — в журнале, запись в девятый раз: не держится; гипотеза
# отклонена как устройство. Числа — ниже, в самом чтении.
#
# Запуск:
#
#     julia julia/kuramoto/counting.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6
const K = 4        # сколько уровней у вибрации
const TOL = 0.02   # что считается «нашлось»: доля расстройки уровня

level_weights(K::Int) = [1.0 / (k * j) for k in 1:K, j in 1:K]

"""
    level_overlap(d, level, band, relative)

Отклик уровня на совпадение. По частоте — как в basic.jl: S(d) ловит разность
k·ωᵢ − l·ωⱼ в единицах частоты. По отношению — S ловит ту же разность в долях
самого уровня: |k·ωᵢ − l·ωⱼ| / (k·ωᵢ).

Разница не в мере, а в том, что слышно. По частоте с ростом уровня совпадение
уходит из полосы: 3·ω₇ и 7·ω₃ при промахе 1 % расходятся на 0.14 — вне полосы
0.05, — а мелкие чужие пары остаются слышны. По отношению уровень слышен
независимо от номера. Это то различие, о котором писал modes.jl: обертоны
стоят далеко друг от друга по частоте, а по отношению — на одном месте.
"""
function level_overlap(d::Float64, level::Float64, band::Float64,
                       relative::Bool)
    relative || return overlap(d, band)
    level == 0.0 && return 0.0
    return 1 / (1 + (d / (band * abs(level)))^2)
end

"""
    step_counted!(medium, dt, strength, C, link, eps = 0.0; relative = false)

Шаг среды, у которой обертонов не написано: у вибрации одна частота, а её
уровни k·ω — чтение того же числа.

Два слагаемых. Первое — прежняя связь между вибрациями, как в `step!`.
Второе — счёт: уровень k вибрации i против уровня j вибрации j', и если
k·ωᵢ ≈ j·ωⱼ', то вибрация i тянется к тому, что этот уровень означает,
то есть к ωᵢ = (j/k)·ωⱼ'. Никакого отношения не задано: заданы только
уровни, а уровни — это ω, взятая целое число раз.

`relative = false` — уровни слышат друг друга по частоте (полоса как в
basic.jl); `relative = true` — по отношению. Оба чтения здесь и стоят.

След (`eps > 0`) — тот же, что в `step_trace!` при `work = true`: помнится
работа уровня, то есть стоял ли он близко.
"""
function step_counted!(m::Medium, dt::Float64, strength::Float64,
                       C::Matrix{Float64}, link::Matrix{Float64},
                       eps::Float64 = 0.0; relative::Bool = false)
    vibs = size(m.omega, 1)
    Kc = size(C, 1)
    drive = zeros(vibs)
    for i in 1:vibs, j in 1:vibs
        j == i && continue
        s = overlap(m.omega[j, 1] - m.omega[i, 1], m.band)
        drive[i] += m.coupling[i, j] * s * sin(m.theta[j, 1] - m.theta[i, 1])
    end
    for i in 1:vibs, j in 1:vibs
        j == i && continue
        for k in 1:Kc, l in 1:Kc
            w = strength * C[k, l] * link[k, l]
            w == 0.0 && continue
            s = level_overlap(k * m.omega[i, 1] - l * m.omega[j, 1],
                              k * m.omega[i, 1], m.band, relative)
            drive[i] += w * s * (l * m.omega[j, 1] / k - m.omega[i, 1])
        end
    end
    # Строя здесь нет и быть не может: он тянет моды p > 1 к p-й доле
    # первой, а у вибрации мода одна — сама она. λ не читается вовсе.
    m.omega .+= dt .* (drive .- m.damping .* (m.omega .- m.omega0))
    m.theta .+= dt .* m.omega
    if eps != 0.0
        for k in 1:Kc, l in 1:Kc
            acc, n = 0.0, 0
            for i in 1:vibs, j in 1:vibs
                j == i && continue
                acc += level_overlap(k * m.omega[i, 1] - l * m.omega[j, 1],
                                     k * m.omega[i, 1], m.band, relative)
                n += 1
            end
            n == 0 && continue
            link[k, l] += dt * eps * (acc / n - link[k, l])
        end
    end
    return m
end

"""
    advance_counted!(medium, seconds, strength, C, link, eps = 0.0; relative = false)

Прожить средой со счётом указанное время.
"""
function advance_counted!(m::Medium, seconds::Float64, strength::Float64,
                          C::Matrix{Float64}, link::Matrix{Float64},
                          eps::Float64 = 0.0; relative::Bool = false)
    for _ in 1:round(Int, seconds / DT)
        step_counted!(m, DT, strength, C, link, eps; relative = relative)
    end
    return m
end

# --- что можно увидеть в описанном ---------------------------------------

"Среда со счётом: одна частота на вибрацию, обертонов не написано."
counted(seed::Int; tuning::Float64 = 0.0, band::Float64 = 0.05,
        link::Float64 = 0.0) =
    build(modes = 1, link = link, band = band, tuning = tuning,
          memory = 0.0, damping = 0.8, seed = seed)

"Частоты стоят как 1 : 2 : 3 … с промахом: аккорд уже почти есть, обертонов нет."
function chord_base!(m::Medium, spread::Float64, seed::Int = 11)
    rng = MersenneTwister(seed)
    base = m.omega0[1, 1]
    for i in 1:size(m.omega, 1)
        m.omega0[i, 1] = base * i * (1 + spread * randn(rng))
    end
    m.omega .= m.omega0
    return m
end

"Разбросать частоты по диапазону: ни к чему не кратные, аккорда не строят."
function scatter_base!(m::Medium, low::Float64, high::Float64, seed::Int = 7)
    rng = MersenneTwister(seed)
    for i in 1:size(m.omega, 1)
        m.omega0[i, 1] = low + (high - low) * rand(rng)
    end
    m.omega .= m.omega0
    return m
end

"Худший промах от целых кратных: насколько вибрации ушли от ωᵢ = i · ω₁."
function chord_error_over(m::Medium)
    worst = 0.0
    for i in 2:size(m.omega, 1)
        worst = max(worst, abs(m.omega[i, 1] / (i * m.omega[1, 1]) - 1))
    end
    return worst
end

"""
    found_pairs(medium, K, tol)

Что счёт нашёл между вибрациями: по каждой паре — уровни (k, l), выполненные
точнее tol. Пара (k, l) значит, что отношение частот равно l / k.
"""
function found_pairs(m::Medium, K::Int, tol::Float64)
    counts = Dict{Tuple{Int, Int}, Int}()
    vibs = size(m.omega, 1)
    pairs, close = 0, 0
    for i in 1:vibs, j in (i + 1):vibs
        pairs += 1
        best, bk, bl = Inf, 1, 1
        for k in 1:K, l in 1:K
            e = abs(k * m.omega[i, 1] - l * m.omega[j, 1]) / (k * m.omega[i, 1])
            if e < best
                best, bk, bl = e, k, l
            end
        end
        if best < tol
            close += 1
            counts[(bk, bl)] = get(counts, (bk, bl), 0) + 1
        end
    end
    return close, pairs, counts
end

"Счёт уровней в строку: (k, l) значит ωᵢ / ωⱼ = l / k."
function found_line(counts::Dict{Tuple{Int, Int}, Int})
    isempty(counts) && return "ничего"
    items = sort(collect(counts), by = x -> (-x[2], x[1]))
    return join([@sprintf("%d:%d ×%d", l, k, n) for ((k, l), n) in items], "  ")
end

"Сбить обертоны со своих мест — как в нынешней записи: они написаны числом."
function detune_ob!(m::Medium, spread::Float64, seed::Int)
    rng = MersenneTwister(seed)
    for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
        m.omega0[i, p] *= 1 + spread * randn(rng)
    end
    m.omega .= m.omega0
    return m
end

"Где осел след: наибольшие веса уровней. (k, l) — это уровни, не отношение."
function link_top(link::Matrix{Float64}, n::Int)
    Kc = size(link, 1)
    items = [(link[k, l], k, l) for k in 1:Kc for l in 1:Kc]
    sort!(items, by = x -> -abs(x[1]))
    return join([@sprintf("(%d,%d) %.2f", k, l, v) for (v, k, l) in items[1:n]], "  ")
end

"""
    main()

Прочитать проверку: держится ли аккорд, когда обертоны не написаны, а сосчитаны.

1. Шаг проверки, как он назван: обертонов нет — что тогда держится.
2. Почти целое: держит ли счёт то, что почти стоит.
3. Нецелое: строит ли счёт аккорд из ничего.
4. След: сила или множитель при силе.
5. То же место в нынешней записи: обертон написан числом, и его толкают.
"""
function main()
    @printf("счёт: у вибрации одна частота, уровней %d, шаг описания %.2f с\n\n", K, DT)

    println("--- шаг проверки: обертонов не написано " * "-"^16)
    println("   У вибрации одна своя частота, уровни k·ω — чтение того же числа.")
    println("   Что тогда держится при λ = 0.")
    m = counted(1)
    chord_base!(m, 0.01, 11)
    inner = maximum(abs(k * m.omega[i, 1] / (k * m.omega[i, 1]) - 1)
                    for i in 1:N, k in 1:K)
    @printf("   промах обертона внутри          %.1e   (арифметика: сравнивать не с чем)\n",
            inner)
    @printf("   промах кратных между вибрациями %.4f   (а это уже состояние)\n",
            chord_error_over(m))
    was = counted(1, tuning = 8.0)
    chord_base!(was, 0.01, 11)
    now = counted(1, tuning = 0.0)
    chord_base!(now, 0.01, 11)
    advance_counted!(was, 60.0, 8.0, level_weights(K), ones(K, K))
    advance_counted!(now, 60.0, 8.0, level_weights(K), ones(K, K))
    @printf("   λ = 8 против λ = 0: частоты     %.1e   разброс %.4f / %.4f\n",
            maximum(abs.(was.omega .- now.omega)), resonance(was), resonance(now))
    println("   Первый промах нулевой потому, что обертон и есть то же число: это")
    println("   не измерение, а арифметика, и держать в описании со счётом нечего —")
    println("   названная проверка пуста по построению. Спрашивать у счёта есть что")
    println("   только между вибрациями: там промах 1 % был, и там его можно")
    println("   потерять.")

    for (relative, hearing) in ((false, "по частоте"), (true, "по отношению"))
        for kk in (4, 10)
            println("\n--- почти целое счёт держит: $(hearing), шкала $(kk) " * "-"^8)
            println("   Частоты стоят как 1 : 2 : 3 … с промахом 1% — на языке счёта")
            println("   это и есть аккорд. λ = 0, память 0, связь — только счёт.")
            for strength in (0.0, 1.0, 4.0, 8.0, 16.0)
                res = [(begin
                    mm = counted(s)
                    chord_base!(mm, 0.01, 11 + s)
                    advance_counted!(mm, 60.0, strength, level_weights(kk),
                                     ones(kk, kk); relative = relative)
                    (chord_error_over(mm), resonance(mm), coherence(mm))
                end) for s in SEEDS]
                @printf("   сила %4.1f   промах %.4f … %.4f   разброс %.3f   согласованность %.3f\n",
                        strength, minimum(r[1] for r in res), maximum(r[1] for r in res),
                        mean(r[2] for r in res), mean(r[3] for r in res))
            end
        end
    end
    println("   При силе 0 счёт не шагает вовсе, и промах остаётся тем, что был.")
    println("   С силой промах растёт, а не падает, — в обоих слушаниях и при любой")
    println("   шкале. Счёт не различает, какое совпадение верное: близких чужих")
    println("   пар больше, чем одной точной, и в сумме они перевешивают. По")
    println("   отношению хуже всего: там среда не расстраивается, а сходится в")
    println("   одну частоту — разброс падает в разы. Аккорд счёт не держит; он")
    println("   его портит, и это не выбор полосы и не выбор шкалы.")

    println("\n--- нецелое счёт не строит " * "-"^21)
    println("   Частоты разбросаны 0.6 … 1.7, ни к чему не кратные: аккорда нет.")
    println("   Что найдёт счёт и во что сведёт среду — в обоих слушаниях.")
    for (relative, hearing) in ((false, "по частоте"), (true, "по отношению"))
        for strength in (0.0, 1.0, 8.0, 16.0)
            res = [(begin
                mm = counted(s)
                scatter_base!(mm, 0.6, 1.7, 700 + s)
                advance_counted!(mm, 60.0, strength, level_weights(K), ones(K, K);
                                 relative = relative)
                close, pairs, counts = found_pairs(mm, K, TOL)
                (close, pairs, counts, resonance(mm), coherence(mm))
            end) for s in SEEDS]
            total = Dict{Tuple{Int, Int}, Int}()
            for r in res, (key, n) in r[3]
                total[key] = get(total, key, 0) + n
            end
            @printf("   %-13s сила %4.1f   нашлось %3d из %3d   разброс %.3f … %.3f   %s\n",
                    hearing, strength, sum(r[1] for r in res), sum(r[2] for r in res),
                    minimum(r[4] for r in res), maximum(r[4] for r in res),
                    found_line(total))
        end
    end
    println("   Счёт находит и здесь — но находит совпадение, а не аккорд: уровень")
    println("   1:1 доступен всегда, поэтому чем сильнее счёт, тем теснее среда")
    println("   сходится в одну частоту. Из нецелого счёт делает не целое, а общее.")

    println("\n--- след: сила или множитель " * "-"^21)
    println("   Собрать среду счётом, дать навести след (память 0.5, 240 с), затем")
    println("   убрать силу и спросить, что осталось.")
    for (relative, hearing) in ((false, "по частоте"), (true, "по отношению"))
        for (weights, name) in ((level_weights(K), "1/(k·l)"), (ones(K, K), "ровный  "))
            res = [(begin
                mm = counted(s)
                scatter_base!(mm, 0.6, 1.7, 700 + s)
                link = ones(K, K)
                advance_counted!(mm, 240.0, 16.0, weights, link, 0.5;
                                 relative = relative)
                was = found_pairs(mm, K, TOL)[1]
                advance_counted!(mm, 60.0, 0.0, weights, link, 0.5;
                                 relative = relative)
                (was, found_pairs(mm, K, TOL)[1], resonance(mm), link_top(link, 3))
            end) for s in SEEDS]
            @printf("   %-13s вес %s   нашлось при силе %2d … %2d, после уборки силы %2d … %2d\n",
                    hearing, name, minimum(r[1] for r in res), maximum(r[1] for r in res),
                    minimum(r[2] for r in res), maximum(r[2] for r in res))
            @printf("%28s разброс после уборки %.3f … %.3f\n", "",
                    minimum(r[3] for r in res), maximum(r[3] for r in res))
            @printf("%28s след: %s\n", "", res[1][4])
        end
    end
    println("   След — не сила, а множитель при силе: сила убрана, и осталось ровно")
    println("   то, что было без неё. Зато след видит, что стояло: в нём остались")
    println("   уровни, через которые счёт шёл, а не те, через которые не шёл.")

    println("\n--- то же место в нынешней записи: обертон написан числом " * "-"^0)
    println("   Тот же вопрос — «держится ли аккорд при λ = 0» — у аккорда, где")
    println("   обертоны написаны числом, как в basic.jl.")
    fresh = build(modes = 3, link = 0.6, band = 0.1, tuning = 0.0, memory = 0.0, seed = 1)
    @printf("   записано, сразу                  %.4f\n", tuning_error(fresh, 1))
    det = build(modes = 3, link = 0.6, band = 0.1, tuning = 0.0, memory = 0.0, seed = 1)
    detune_ob!(det, 0.08, 901)
    @printf("   записано и сбито на 8%%           %.4f\n", tuning_error(det, 1))
    free = build(modes = 3, link = 0.6, band = 0.1, tuning = 0.0, memory = 0.0, seed = 1)
    advance!(free, 60.0)
    @printf("   записано, минута при λ = 0       %.4f\n", tuning_error(free, 1))
    held = build(modes = 3, link = 0.6, band = 0.1, tuning = 8.0, memory = 0.0, seed = 1)
    advance!(held, 60.0)
    @printf("   записано, минута при λ = 8       %.4f\n", tuning_error(held, 1))

    println("\n   толчок обертону 0.3, затем минута:")
    for tuning in (0.0, 8.0)
        mm = build(modes = 3, link = 0.6, band = 0.1, tuning = tuning,
                   memory = 0.0, seed = 1)
        advance!(mm, 60.0)
        perturb!(mm, 1, 0.3, 2)
        advance!(mm, 60.0)
        @printf("   λ = %3.1f   отклонение второй моды %.3f   промах кратных %.4f\n",
                tuning, drift(mm, 1, 2), tuning_error(mm, 1))
    end
    println("   Толчок обертона в нынешней записи — событие с числом. В описании со")
    println("   счётом такого толчка некуда отнести: второй моды нет как числа.")
    println("   Разница не в числах, а в том, что можно сказать: счёт молчит о")
    println("   нецелом, а запись нет.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


