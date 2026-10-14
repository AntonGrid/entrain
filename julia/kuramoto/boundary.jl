# Entrain. Где кончается среда.
#
# Это проверка, а не описание. Уравнения те же, что в basic.jl, а вопрос один:
# что делает среду средой — место, где кончаются узлы, или совпадение.
#
# В книге сказано: entrain локален по природе, глобальной среды не бывает; у
# среды есть края, и что там происходит, в уравнениях пока нет. Здесь
# спрашивается ровно это.
#
# Сцена одна и та же во всех чтениях: среда, у которой край есть, узел на краю
# и узел вне среды. Мера одна: идёт ли вибрация вместе с целым — ⟨cos(θᵢ − Ψ)⟩,
# где Ψ — фаза среды, — и куда пришла её частота.
#
# Три чтения:
#
#     край       — у узла на краю связей вдвое меньше. Он всё ещё среда?
#     вне        — узел пришит к среде одной нитью; его своя частота отстоит.
#     шире полосы — среда, чей собственный разброс не влезает в полосу.
#
# Память здесь выключена (ε = 0): смотрим не след, а границу.
#
# Запуск:
#
#     julia julia/kuramoto/boundary.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6

# --- прибор ---------------------------------------------------------------

"""
    together!(medium, body, seconds)

Прожить средой ещё seconds и вернуть по каждой вибрации ⟨cos(θᵢ − Ψ)⟩ — идёт ли
она вместе с целым. Ψ — фаза среды по вибрациям body. Ноль значит «живёт сама»,
единица — «в фазе с целым».
"""
function together!(m::Medium, body, seconds::Float64)
    acc = zeros(size(m.theta, 1))
    steps = round(Int, seconds / DT)
    for _ in 1:steps
        step!(m, DT)
        psi = angle(mean(exp.(im .* m.theta[body, 1])))
        acc .+= cos.(m.theta[:, 1] .- psi)
    end
    return acc ./ steps
end

"""
    strand(; reach = 1, tie = 0.0, gap = 0.0, band = Inf, spread = 0.03, seed = 1)

Среда-цепочка из двенадцати вибраций и гость — тринадцатая.

Связь в среде есть только с ближними на расстоянии не больше reach: у края её
поэтому вдвое меньше, чем внутри (при reach = 3 — три из шести). Это и есть
край как место. Гость пришит одной нитью к краю среды, сила нити tie, его своя
частота отстоит от общего звука среды на gap.

Возвращает три числа: ⟨cos⟩ по всем вибрациям; куда пришла частота гостя
(0 — остался собой, 1 — стал общим звуком); согласованность среды.
"""
function strand(; reach::Int = 1, tie::Float64 = 0.0, gap::Float64 = 0.0,
                band::Float64 = Inf, spread::Float64 = 0.03, seed::Int = 1,
                warming::Float64 = 120.0, reading::Float64 = 30.0)
    n, t = 12, 13
    rng = MersenneTwister(seed)
    omega0 = zeros(t, 1)
    omega0[1:n, 1] = 1.0 .+ spread .* randn(rng, n)
    omega0[t, 1] = mean(omega0[1:n, 1]) + gap
    theta = 2π .* rand(rng, t, 1)
    W = zeros(t, t)
    for i in 1:n, j in 1:n
        d = abs(i - j)
        (d == 0 || d > reach) && continue
        W[i, j] = 0.6 / (2 * reach)
    end
    W[t, n] = tie
    W[n, t] = tie
    m = Medium(omega0 = omega0, theta = theta, coupling = W, band = band)
    advance!(m, warming)
    here = mean(m.omega[1:n, 1])
    taken = (m.omega[t, 1] - m.omega0[t, 1]) / (here - m.omega0[t, 1])
    return together!(m, 1:n, reading), taken, coherence(m)
end

"""
    unison(; band = Inf, spread = 0.15, seed = 1)

Среда без гостя: десять вибраций, связь полная (как в build), разброс своих
частот spread. Возвращает согласованность и разброс частот через минуту.
"""
function unison(; band::Float64 = Inf, spread::Float64 = 0.15, seed::Int = 1)
    n = 10
    rng = MersenneTwister(seed)
    om = 1.0 .+ spread .* randn(rng, n)
    th = 2π .* rand(rng, n)
    m = Medium(omega0 = reshape(om, n, 1), theta = reshape(copy(th), n, 1),
               coupling = fill(0.6 / n, n, n), band = band)
    advance!(m, 60.0)
    return coherence(m), resonance(m)
end

const TIES = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)

"""
    needed(; band = Inf, gap = 0.05, warming = 120.0)

Наименьшая нить, при которой узел входит в среду: середина шести начал даёт не
меньше половины пути к общему звуку. Если и самая большая нить не вводит —
возвращается строка «> 3.2»: это больше, чем все связи самой среды.
"""
function needed(; band::Float64 = Inf, gap::Float64 = 0.05, warming::Float64 = 120.0)
    for tie in TIES
        v = sort([strand(tie = tie, gap = gap, band = band,
                         warming = warming, seed = s)[2] for s in SEEDS])
        (v[3] + v[4]) / 2 >= 0.5 && return @sprintf("%.2f", tie)
    end
    return "> $(TIES[end])"
end


# --- чтение ---------------------------------------------------------------

"""
    scene_shore()

Край как место. У узла на краю связей вдвое (втрое, вчетверо) меньше, чем
внутри. Печатается, идёт ли он вместе с целым — по вибрациям 1, 6 и 12.
"""
function scene_shore()
    println("--- край: связей вдвое меньше — выходит ли узел из среды " * "-"^13)
    println("   ⟨cos(θ − Ψ)⟩: 1 — в фазе с целым, 0 — живёт сама; шесть начал,")
    println("   прогрев 120 с (то же и за 2400 с: разрыв не растёт, край не выпадает)")
    @printf("   %-14s %-13s %-13s %-13s %-13s\n", "связей у края", "край 1",
            "середина 6", "край 12", "гость без связи")
    for reach in (1, 2, 3, 6)
        rows = [strand(reach = reach, seed = s) for s in SEEDS]
        lo(i) = minimum(r[1][i] for r in rows)
        hi(i) = maximum(r[1][i] for r in rows)
        @printf("   %-14s %.3f…%.3f  %.3f…%.3f  %.3f…%.3f  %.3f…%.3f\n",
                "$(reach) из $(2 * reach)", lo(1), hi(1), lo(6), hi(6), lo(12), hi(12),
                lo(13), hi(13))
    end
    println("   Край от середины не отличается: при связях 2 из 4 и больше числа")
    println("   совпадают, при одной из двух край сходится медленнее, но не выпадает.")
    println("   Край как место границы среды не делает.")
    println("   Гость без связи (последний столбец) даёт любое число — почти от −1")
    println("   до 1: его ⟨cos⟩ задан начальным сдвигом фаз, а не средой. У связанного")
    println("   узла он ниже 0.59 не опускается: среду видно по тому, что ⟨cos⟩ держится.")
    return nothing
end

"""
    scene_outside()

Узел вне среды. Он не в цепочке, а рядом с её краем, и пришит одной нитью той
же силы, что и сам край (0.3 — одна связь из двух; у внутреннего узла две
связи, то есть 0.6). От края он отличается одним: его своя частота отстоит от
общего звука среды. Печатается, куда пришла его частота, и сколько нити нужно,
чтобы он вошёл.
"""
function scene_outside()
    println("\n--- вне: та же тяга, что у края, — другая частота " * "-"^16)
    println("   доля входа гостя при нити 0.3 (как у края): 0 — остался собой,")
    println("   1 — стал общим звуком; наименьшее … наибольшее по шести началам")
    @printf("   %-10s", "полоса Π")
    for gap in (0.05, 0.1, 0.2, 0.5)
        @printf("%16s", "разрыв $gap")
    end
    println()
    for band in (Inf, 0.5, 0.1, 0.02)
        @printf("   %-10s", band == Inf ? "∞" : string(band))
        for gap in (0.05, 0.1, 0.2, 0.5)
            rows = [strand(tie = 0.3, gap = gap, band = band, seed = s) for s in SEEDS]
            lo = minimum(r[2] for r in rows)
            hi = maximum(r[2] for r in rows)
            @printf("%16s", @sprintf("%.2f…%.2f", lo, hi))
        end
        println()
    end

    println("\n   сколько нити нужно, чтобы узел вошёл (середина шести начал ≥ 0.5)")
    @printf("   %-10s", "полоса Π")
    for gap in (0.05, 0.1, 0.2, 0.5)
        @printf("%16s", "разрыв $gap")
    end
    println()
    for band in (Inf, 0.5, 0.1, 0.02)
        @printf("   %-10s", band == Inf ? "∞" : string(band))
        for gap in (0.05, 0.1, 0.2, 0.5)
            @printf("%16s", needed(band = band, gap = gap))
        end
        println()
    end

    println("\n   и то же за 2400 с — не медлит ли узел (Π = 0.1, разрыв 0.5, нить 0.3):")
    brief = [strand(tie = 0.3, gap = 0.5, band = 0.1, warming = 120.0, seed = s)[2]
             for s in SEEDS]
    long = [strand(tie = 0.3, gap = 0.5, band = 0.1, warming = 2400.0, seed = s)[2]
            for s in SEEDS]
    @printf("   за 120 с %.2f…%.2f, за 2400 с %.2f…%.2f\n",
            minimum(brief), maximum(brief), minimum(long), maximum(long))

    println("   Пока разрыв не больше пары полос, нить нужна та же, что без полосы:")
    println("   при разрыве 0.05 … 0.2 порог 0.05 … 0.20. Дальше он растёт обвально:")
    println("   при Π = 0.1 и разрыве 0.5 (пять полос) нужна нить 1.6 … 3.2 — больше")
    println("   всех связей самой среды (0.3 у края, 0.6 внутри); при Π = 0.02 и")
    println("   разрыве 0.5 не берётся и 3.2. Число гуляет от начальных условий, а")
    println("   порядок нет. И ждать нечего: за 2400 с — то же самое.")
    println("   Край с той же тягой 0.3 — в среде, а узел вне её — не в среде: они")
    println("   отличаются только совпадением частот. Полоса — не помеха связи;")
    println("   она и есть граница среды.")
    return nothing
end

"""
    scene_wider()

Среда шире полосы. Та же среда без гостя: разброс её собственных частот против
полосы отклика. Печатается согласованность через минуту.
"""
function scene_wider()
    println("\n--- шире полосы: среда, чей разброс в полосу не влезает " * "-"^13)
    @printf("   %-12s %-16s %-16s %-16s\n", "полоса Π",
            "разброс 0.02", "разброс 0.05", "разброс 0.15")
    for band in (Inf, 0.5, 0.2, 0.1, 0.05, 0.02)
        @printf("   %-12s", band == Inf ? "∞" : string(band))
        for spread in (0.02, 0.05, 0.15)
            rows = [unison(band = band, spread = spread, seed = s) for s in SEEDS]
            lo = minimum(r[1] for r in rows)
            hi = maximum(r[1] for r in rows)
            @printf(" %-16s", @sprintf("%.3f…%.3f", lo, hi))
        end
        println()
    end
    println("   Среда стоит, пока её собственный разброс укладывается в полосу:")
    println("   разброс 0.15 держится при Π = 0.2 по всем началам и уже не по всем")
    println("   при 0.1 (в худшем начале 0.66 … 0.75); разброс 0.05 держится при 0.05,")
    println("   а при 0.02 — на грани: в одних начальных условиях стоит, в других")
    println("   рассыпается. Границу среды делает она же — полоса, которой среда слышит;")
    println("   отдельного края у среды нет.")
    return nothing
end

"""
    boundary()

Прочитать описание: край, узел вне среды, среда шире полосы.
"""
function boundary()
    @printf("среда из вибраций, шаг описания %.2f с — где кончается среда\n\n", DT)
    scene_shore()
    scene_outside()
    scene_wider()
    println("\n   Одно: границу среды делает полоса отклика, а не место. Край, где")
    println("   связей вдвое меньше, — та же среда; узел вне среды входит в неё при")
    println("   той же связи, что у края, если совпал по частоте, и не входит ни при")
    println("   какой, если вышел за полосу. Та же полоса держит и саму среду: чей")
    println("   разброс шире полосы — та рассыпается. Отдельного края у среды нет:")
    println("   среда кончается там, где она перестаёт отзываться.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    boundary()
end
