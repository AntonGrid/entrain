# Entrain. Курамото: десять вибраций.
#
# Это описание, а не проверка.
# Здесь нет порогов, гипотез, вердиктов и «а что если».
#
# Записи:
#
#     состояние — частота; у одной вибрации их несколько: аккорд
#     связь     — coupling: как одна вибрация тянет другую
#     внутри    — inside: среда внутри вибрации; та же среда, взятая близко:
#                 её вибрации — моды этой вибрации, и она у каждой своя
#     резонанс  — отклик по совпадению частот: полоса Π широкая — связь
#                 отзывается на многое, узкая — только на совпадающее
#     строй     — жёсткость аккорда: держит обертоны в их отношении
#     затухание — возвращение к своим частотам
#     память    — деформация связи тем, что совпало (русло) — и внутри тоже;
#                 срок памяти 1/ε у каждой связи свой (εᵢⱼ), а не один на уровень
#
#     θ̇ᵢₘ = ωᵢₘ
#     ω̇ᵢₘ = Σⱼₙ Wᵢⱼ S(ωⱼₙ − ωᵢₘ) sin(θⱼₙ − θᵢₘ)
#                 + Σ_q Uᵢₘq S(ωᵢ_q − ωᵢₘ) sin(θᵢ_q − θᵢₘ)
#                 − λ (ωᵢₘ − m ωᵢ₁) − γ (ωᵢₘ − ω₀ᵢₘ)
#     Ẇᵢⱼ = εᵢⱼ ( Σₚₙ S cos(θᵢₚ − θⱼₙ) / Σₚₙ S − Wᵢⱼ )
#     U̇ᵢₘq = εᵤᵢₘq ( cos(θᵢₘ − θᵢ_q) − Uᵢₘq )    — у каждой вибрации своё русло
#     S(Δ) = 1 / (1 + (Δ/Π)²)
#
# Сила входит в частоту, а не в фазу: фаза интегрирует частоту. Порядок шага —
# в `step!`: частота обновляется первой, фаза берёт уже новую частоту, связь
# пишется последней. В пределе dt → 0 это θ̇ = ω, ω̇ = F.
#
# При одной моде на вибрацию и Π = ∞ это в точности прежние уравнения:
# у вибрации была одна частота, отклик не зависел от совпадения.
# Внутренняя связь Uᵢ видна, только когда мод больше одной; при Uᵢ = 0
# прежнее чтение не меняется ни на число.
#
# εᵢⱼ — срок памяти связи, а не уровня: у каждой связи он свой, поэтому
# «мгновенное» и «годовое» стоят рядом в одной среде и не спорят. Задаётся
# раскладкой (`memory_matrix`); когда раскладки нет, у уровня одно ε
# (`memory`) и запись прежняя до последнего бита — раскладка не второе число,
# а то же самое, разложенное по связям. То же у мира внутри вибрации
# (`inner_memory_matrix`).
#
# Было приближение: связь внутри вибрации одна на все вибрации сразу, и наводилась
# она тем, что совпало у них у всех. Здесь она у каждой своя — среда внутри
# вибрации есть та же среда, и русло у неё своё.
#
# Запуск:
#
#     julia julia/kuramoto/basic.jl

using Printf
using Random
using Statistics

# Сколько вибраций в среде. Десять — паспорт описания; `ENTRAIN_N` задаёт то же
# число за границей паспорта, не меняя запись: связь на пару остаётся `link / N`,
# поэтому полная тяга среды та же.
const N = parse(Int, get(ENV, "ENTRAIN_N", "10"))
const DT = 0.01     # шаг описания, с

"""
    Medium

Среда — не набор объектов, а взаимодействие вибраций.
У каждой вибрации не одна частота, а несколько: аккорд.
И у каждой вибрации есть свой мир — та же среда, взятая близко (`inside`).

- `omega0`   — свои частоты: `omega0[i, m]` — мода m вибрации i;
- `omega`    — частоты сейчас: это и есть состояние;
- `theta`    — фазы;
- `coupling` — связь между вибрациями: `coupling[i, j]` — насколько j тянет i;
- `band`     — полоса отклика Π: чем уже, тем строже резонанс;
- `tuning`   — λ: строй, жёсткость аккорда: держит обертоны в их отношении
              к основной частоте (`ωᵢₘ → m · ωᵢ₁`); при λ = 0 аккорд не держится.
              Строй — не среда, а форма: он живёт у вибрации, в её мире. Среда
              снаружи его только задаёт — миры берут его себе, и свой строй мира
              среда не перебивает. У вибрации без мира (одна мода) строя нет:
              держать нечего. Проверено чтением «где живёт строй» в `modes.jl`;
- `memory`   — ε: как быстро связь деформируется тем, что через неё прошло;
- `memory_matrix` — ε у каждой связи: `memory_matrix[i, j]` — срок памяти связи
              (i, j) свой. `nothing` — у всего уровня одно ε, и тогда запись
              прежняя ни на число: в шаге то же сложение, меняется только то,
              откуда взят множитель. Так «мгновенное» и «годовое» могут стоять
              рядом в одной среде, а написанное — пережить окно у той связи,
              у которой срок длиннее (чтение «срок у связи» в `terms.jl`);
- `damping`  — γ: как быстро частоты возвращаются к своим;
- `inside`   — миры внутри вибраций: `inside[i]` — среда мод вибрации i, со
              своими связью, полосой, строем, памятью и затуханием; `nothing` —
              у вибрации мира нет.

Среду собирают именованно — `Medium(omega0 = …, theta = …, coupling = …)` для
заданного состояния и `build(…)` для сцены. Позиционной сборки нет нарочно:
новая запись в `Medium` должна называться, а не вставать по счёту, — иначе она
молча сдвинет все числа прежних чтений.
"""
mutable struct Medium{A<:AbstractMatrix{Float64}, B<:AbstractMatrix{Float64},
                      C<:AbstractMatrix{Float64}, D<:AbstractMatrix{Float64}}
    omega0::A
    omega::B
    theta::C
    coupling::D
    band::Float64
    tuning::Float64
    memory::Float64
    memory_matrix::Union{Nothing, Matrix{Float64}}
    damping::Float64
    inside::Union{Nothing, Vector{Medium}}

    function Medium(; omega0::AbstractMatrix{Float64},
                      omega::AbstractMatrix{Float64} = copy(omega0),
                      theta::AbstractMatrix{Float64},
                      coupling::AbstractMatrix{Float64},
                      band::Float64 = Inf, tuning::Float64 = 0.0,
                      memory::Float64 = 0.0,
                      memory_matrix::Union{Nothing, AbstractMatrix{Float64}} = nothing,
                      damping::Float64 = 0.8,
                      inside::Union{Nothing, Vector{Medium}} = nothing)
        size(omega0) == size(omega) == size(theta) ||
            throw(ArgumentError("состояние — одна раскладка: вибрации × моды"))
        size(coupling) == (size(omega, 1), size(omega, 1)) ||
            throw(ArgumentError("связь — между вибрациями: по числу вибраций"))
        memory_matrix === nothing ||
            size(memory_matrix) == size(coupling) ||
            throw(ArgumentError("раскладка памяти — по связям: вибрации × вибрации"))
        return new{typeof(omega0), typeof(omega), typeof(theta), typeof(coupling)}(
            omega0, omega, theta, coupling, band, tuning, memory,
            memory_matrix === nothing ? nothing : Matrix{Float64}(memory_matrix),
            damping, inside)
    end
end

"""
    world(medium, i)

Среда внутри вибрации i — или `nothing`, если у вибрации мира нет. Мир не копия:
он смотрит на те же числа, что и среда снаружи, а его вибрации — моды вибрации i.
"""
world(m::Medium, i::Int) = m.inside === nothing ? nothing : m.inside[i]

"""
    link_memory(medium, i, j)

ε связи вибраций i и j: у каждой связи своё, если среда собрана с раскладкой
(`memory_matrix`); если раскладки нет — у всего уровня одно ε (`memory`).
"""
link_memory(m::Medium, i::Int, j::Int) =
    m.memory_matrix === nothing ? m.memory : m.memory_matrix[i, j]

"""
    has_memory(medium)

Есть ли у среды память: ноль значит, что связи не пишутся вовсе, и шаг памяти
пропускается — так же, как прежде при одном ε = 0.
"""
has_memory(m::Medium) = m.memory_matrix === nothing ? m.memory != 0.0 :
                        any(ε -> ε != 0.0, m.memory_matrix)

"""
    inside!(medium; coupling, band, memory, tuning, damping)

Дать каждой вибрации свой мир: та же среда, взятая близко. Вибрации мира — моды
вибрации; состояние у них общее с внешней средой (мир смотрит на ту же строку, а
не на свою копию), а связь, полоса, строй, память и затухание у мира свои.

По умолчанию мир берёт у среды снаружи полосу, строй и затухание, а память у
него своя и её нет (εᵤ = 0): тогда мир ничего не добавляет к уже сказанному, и
числа не двигаются.

Память у мира устроена как у среды снаружи: `memory` — одно εᵤ на все связи мод,
`memory_matrix` — εᵤ у каждой пары мод своё (`nothing` — одно на все).
"""
function inside!(m::Medium; coupling::AbstractMatrix{Float64},
                  band::Float64 = m.band, memory::Float64 = 0.0,
                  memory_matrix::Union{Nothing, AbstractMatrix{Float64}} = nothing,
                  tuning::Float64 = m.tuning, damping::Float64 = m.damping)
    modes = size(m.omega, 2)
    size(coupling) == (modes, modes) ||
        throw(ArgumentError("связь внутри вибрации — между её модами"))
    memory_matrix === nothing ||
        size(memory_matrix) == (modes, modes) ||
        throw(ArgumentError("раскладка памяти мира — по связям мод: моды × моды"))
    worlds = Vector{Medium}(undef, size(m.omega, 1))
    for i in eachindex(worlds)
        worlds[i] = Medium(omega0 = reshape(view(m.omega0, i, :), modes, 1),
                           omega = reshape(view(m.omega, i, :), modes, 1),
                           theta = reshape(view(m.theta, i, :), modes, 1),
                           coupling = copy(coupling), band = band,
                           tuning = tuning, memory = memory,
                           memory_matrix = memory_matrix === nothing ? nothing :
                                           copy(memory_matrix),
                           damping = damping)
    end
    m.inside = worlds
    return m
end

"""
    overlap(Δω, band)

Отклик по совпадению частот: S(Δ) = 1 / (1 + (Δ/Π)²).
При Δ = 0 отклик полный; дальше падает как 1/Δ². Это и есть резонанс:
среда отзывается на то, что звучит на её частоте, и почти не отзывается
на то, что звучит далеко.
"""
overlap(Δω::Float64, band::Float64) = 1 / (1 + (Δω / band)^2)

"""
    inner_force!(world, drive, row)

Сила, которой мир внутри вибрации тянет свои вибрации — моды вибрации `row`.
Складывается туда же, куда среда сложила силы между вибрациями: вибрация делает
один шаг, а не по шагу на масштаб.
"""
function inner_force!(w::Medium, drive::AbstractMatrix{Float64}, row::Int)
    vibs = size(w.omega, 1)
    for p in 1:vibs, q in 1:vibs
        q == p && continue
        s = overlap(w.omega[q, 1] - w.omega[p, 1], w.band)
        drive[row, p] += w.coupling[p, q] * s * sin(w.theta[q, 1] - w.theta[p, 1])
    end
    return drive
end

"""
    inner_trace!(world, dt)

След внутри вибрации: связь в мире наводится тем, что через неё прошло, — как и
связь между вибрациями, только у каждой вибрации это своё. Совпало — потому что
через эту пару мод прошло именно это. Срок памяти у каждой пары мод свой
(`link_memory`), как и у каждой связи между вибрациями.
"""
function inner_trace!(w::Medium, dt::Float64)
    vibs = size(w.omega, 1)
    for p in 1:vibs, q in 1:vibs
        q == p && continue
        s = overlap(w.omega[q, 1] - w.omega[p, 1], w.band)
        s == 0.0 && continue
        target = cos(w.theta[p, 1] - w.theta[q, 1])
        w.coupling[p, q] += dt * link_memory(w, p, q) * (target - w.coupling[p, q])
    end
    return w
end

"""
    step!(medium, dt)

Один шаг описания: связь двигает частоты, затухание возвращает их к своим,
прошедшее деформирует связь. Внутри вибрации — тот же закон на меньшем
масштабе: моды тянут друг друга, и то, что через них прошло, оставляет след
там же — у каждой вибрации в своём мире.

Вибрация делает один шаг, а не по шагу на масштаб: силы всех масштабов
складываются в одну. Мир внутри вибрации — среда, как и всё здесь, и его можно
шагнуть отдельно (`step!(world(m, i), dt)`) — тогда он живёт сам.
"""
function step!(m::Medium, dt::Float64)
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
    for i in 1:vibs
        w = world(m, i)
        (w === nothing || all(iszero, w.coupling)) && continue
        inner_force!(w, drive, i)
    end
    # Строй и затухание: у кого есть мир — того держит мир, у кого нет — среда.
    for i in 1:vibs, p in 1:modes
        w = world(m, i)
        tuning = w === nothing ? m.tuning : w.tuning
        damping = w === nothing ? m.damping : w.damping
        if p > 1 && tuning != 0.0
            drive[i, p] -= tuning * (m.omega[i, p] - p * m.omega[i, 1])
        end
        damping != 0.0 && (drive[i, p] -= damping * (m.omega[i, p] - m.omega0[i, p]))
    end
    m.omega .+= dt .* drive
    m.theta .+= dt .* m.omega
    if has_memory(m)
        for i in 1:vibs, j in 1:vibs
            i == j && continue
            acc, weight = 0.0, 0.0
            for p in 1:modes, n in 1:modes
                s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
                acc += s * cos(m.theta[i, p] - m.theta[j, n])
                weight += s
            end
            target = weight == 0.0 ? m.coupling[i, j] : acc / weight
            m.coupling[i, j] += dt * link_memory(m, i, j) * (target - m.coupling[i, j])
        end
    end
    for i in 1:vibs
        w = world(m, i)
        (w === nothing || !has_memory(w) || all(iszero, w.coupling)) && continue
        inner_trace!(w, dt)
    end
    return m
end

# --- что можно увидеть в описанном ---------------------------------------

"""
    coherence(medium)

Согласованность среды: звучит ли вся среда как одно.
Единица — все фазы совпали, ноль — фазы разошлись полностью.
"""
coherence(m::Medium) = abs(mean(exp.(im .* m.theta)))

"""
    partial_coherence(medium, p)

Согласованность по одной частоте: сошлись ли между собой p-е моды
разных вибраций. Именно здесь видно резонанс: среда может не сходиться
в одно целое, а сходиться по совпадениям частот.
"""
partial_coherence(m::Medium, p::Int) = abs(mean(exp.(im .* m.theta[:, p])))

"""
    chord(medium, i, t)

Что звучит i-я вибрация в момент t: её аккорд целиком, все моды сразу.
"""
chord(m::Medium, i::Int, t::Float64) = mean(cos.(m.omega[i, :] .* t .+ m.theta[i, :]))

"""
    wave(medium, t)

Что слышно в точке среды: наложение всех мод всех вибраций.
"""
wave(m::Medium, t::Float64) = mean(cos.(m.omega .* t .+ m.theta))

"""
    resonance(medium)

Разброс частот по всей среде: чем ближе частоты, тем уже резонанс.
"""
resonance(m::Medium) = maximum(m.omega) - minimum(m.omega)

"""
    spread(medium, i)

Разброс частот внутри одной вибрации: ширина её аккорда.
"""
spread(m::Medium, i::Int) = maximum(m.omega[i, :]) - minimum(m.omega[i, :])

"""
    unison(medium, i)

Насколько аккорд i-й вибрации сжался: ноль — ширина та же, что была,
единица — все моды сошлись в одну частоту. Видно, гасит ли внутренняя
связь различие между модами.
"""
function unison(m::Medium, i::Int)
    was = maximum(m.omega0[i, :]) - minimum(m.omega0[i, :])
    was == 0.0 && return 0.0
    return 1 - spread(m, i) / was
end

"""
    coincidence(medium, i)

Совпадение фаз внутри вибрации: среднее косинуса разности её фаз по парам
мод. Единица — моды идут в фазе, ноль — разошлись, минус единица — в
противофазе. Это ровно та мера, которую пишет русло внутри вибрации
(`inner_trace!`): русло идёт к совпадению, а не к отношению.
"""
function coincidence(m::Medium, i::Int)
    modes = size(m.theta, 2)
    modes < 2 && return 0.0
    acc = 0.0
    for p in 1:modes, q in 1:modes
        p == q && continue
        acc += cos(m.theta[i, p] - m.theta[i, q])
    end
    return acc / (modes * (modes - 1))
end

"""
    inner_links(medium)

Связь внутри каждой вибрации своим рядом: у каждой вибрации свой мир, и своё
русло в нём. Пустой ряд значит, что миров нет — у вибрации меньше двух мод.
"""
function inner_links(m::Medium)
    links = Float64[]
    m.inside === nothing && return links
    for w in m.inside
        modes = size(w.coupling, 1)
        modes < 2 && continue
        s = 0.0
        for p in 1:modes, q in 1:modes
            p == q && continue
            s += w.coupling[p, q]
        end
        push!(links, s / (modes * (modes - 1)))
    end
    return links
end

"""
    inner_link(medium)

Средняя связь внутри вибрации — по мирам: у каждой вибрации свой мир, и своё
русло в нём. Это то, что навела внутренняя среда.
"""
function inner_link(m::Medium)
    links = inner_links(m)
    return isempty(links) ? 0.0 : sum(links) / length(links)
end

"""
    tuning_error(medium, i)

Насколько строй i-й вибрации сошёл: отношение обертонов к основной частоте.
Ноль — аккорд держится ровно, как задан.
"""
function tuning_error(m::Medium, i::Int)
    modes = size(m.omega, 2)
    modes == 1 && return 0.0
    base = m.omega[i, 1]
    return maximum(abs(m.omega[i, p] - p * base) for p in 2:modes)
end

"""
    perturb!(medium, i, shift, p = 1)

Событие: толчок. Меняет частоту моды p вибрации i. Среда ничего об этом
не знает — она просто становится другой.
"""
perturb!(m::Medium, i::Int, shift::Float64, p::Int = 1) = (m.omega[i, p] += shift; m)

"""
    drift(medium, i, p = 1)

Отклонение: насколько частота моды ушла от своей.
"""
drift(m::Medium, i::Int, p::Int = 1) = abs(m.omega[i, p] - m.omega0[i, p])

# --- построение -----------------------------------------------------------

"""
    build(; modes = 1, link = 0.6, band = Inf, tuning = 0.0,
            memory = 1e-3, inner = 0.0, inner_memory = 0.0,
            inner_matrix = nothing, damping = 0.8, seed = 4)

Построить среду: N вибраций, у каждой modes частот и свой мир внутри.

При `modes = 1` и `band = Inf` это прежнее описание: одна частота на вибрацию,
отклик без различия совпадений. Аккорд, полоса и строй — тот же самый язык,
просто теперь в нём есть то, что слово называло аккордом.

`inner` — связь внутри вибрации: то же, что `link`, но на меньшем масштабе, и
своя у каждой вибрации. Своя матрица (`inner_matrix`) задаётся, когда связь
внутри должна зависеть от того, какие это моды, а не только от того, насколько
они совпали; каждому миру достаётся её копия. `inner_memory` — εᵤ: русло внутри,
то же, что `memory`, но на меньшем масштабе, и тоже своё у каждой вибрации.
"""
function build(; modes::Int = 1, link::Float64 = 0.6, band::Float64 = Inf,
                 tuning::Float64 = 0.0, memory::Float64 = 1e-3,
                 memory_matrix::Union{Nothing,AbstractMatrix{Float64}} = nothing,
                 inner::Float64 = 0.0, inner_memory::Float64 = 0.0,
                 inner_matrix::Union{Nothing,Matrix{Float64}} = nothing,
                 inner_memory_matrix::Union{Nothing,AbstractMatrix{Float64}} = nothing,
                 damping::Float64 = 0.8, seed::Int = 4)
    modes >= 1 || throw(ArgumentError("у вибрации хотя бы одна частота"))
    rng = MersenneTwister(seed)
    base = 1.0 .+ 0.15 .* randn(rng, N)          # основные частоты вибраций
    omega0 = zeros(N, modes)
    for i in 1:N
        omega0[i, 1] = base[i]
        for p in 2:modes
            omega0[i, p] = base[i] * p * (1 + 0.01 * randn(rng))   # обертоны
        end
    end
    theta = 2π .* rand(rng, N, modes)
    U = inner_matrix === nothing ? fill(inner, modes, modes) : copy(inner_matrix)
    for p in 1:modes
        U[p, p] = 0.0                              # мода не тянет саму себя
    end
    m = Medium(omega0 = omega0, theta = theta, coupling = fill(link / N, N, N),
               band = band, tuning = tuning, memory = memory,
               memory_matrix = memory_matrix, damping = damping)
    modes > 1 && inside!(m, coupling = U, memory = inner_memory,
                         memory_matrix = inner_memory_matrix)
    return m
end

"""
    advance!(medium, seconds)

Прожить средой указанное время.
"""
function advance!(m::Medium, seconds::Float64)
    for _ in 1:round(Int, seconds / DT)
        step!(m, DT)
    end
    return m
end

"""
    show(medium, label)

Показать, в каком состоянии среда.
"""
function show(m::Medium, label::AbstractString)
    @printf("%-28s согласованность %.3f   разброс частот %.3f   связь %.4f\n",
            label, coherence(m), resonance(m), mean(m.coupling))
    modes = size(m.omega, 2)
    if modes > 1
        parts = join([@sprintf("%.3f", partial_coherence(m, p)) for p in 1:modes], "  ")
        @printf("%-28s по совпадениям:              %s\n", "", parts)
    end
    return nothing
end

"""
    report_range(name, values, unit = "")

Наименьшее … наибольшее из ряда чтений при разных начальных условиях.
Одно число — это один бросок; ряд показывает, где свойство, а где случай.
"""
function report_range(name::AbstractString, values, unit::AbstractString = "")
    @printf("   %-34s %8.4f … %-8.4f%s\n", name, minimum(values), maximum(values), unit)
    return nothing
end

# --- чтение ---------------------------------------------------------------

"""
    ranges()

То же чтение, но не одним броском: при шести начальных условиях сразу.
Отдельное число в среде зависит от того, с чего начали; размах — много меньше.
Так видно, что здесь свойство описания, а что — бросок генератора.
"""
function ranges()
    seeds = 1:6
    println("\n--- то же чтение при шести начальных условиях: наименьшее … наибольшее " * "-"^3)

    wake = [(begin
        m = build(seed = s)
        before = (coherence(m), resonance(m))
        advance!(m, 60.0)
        (before..., coherence(m), resonance(m))
    end) for s in seeds]
    report_range("согласованность сразу", [w[1] for w in wake])
    report_range("разброс частот сразу", [w[2] for w in wake])
    report_range("согласованность через минуту", [w[3] for w in wake])
    report_range("разброс частот через минуту", [w[4] for w in wake])

    ping = [(begin
        m = build(seed = s)
        advance!(m, 60.0)
        perturb!(m, 1, 1.5)
        a = drift(m, 1)
        advance!(m, 6.0)
        (a, drift(m, 1))
    end) for s in seeds]
    report_range("отклонение после толчка", [p[1] for p in ping])
    report_range("отклонение через шесть секунд", [p[2] for p in ping])

    held = [(begin
        a = build(modes = 3, band = 0.1, tuning = 8.0, seed = s)
        b = build(modes = 3, band = 0.1, tuning = 0.0, seed = s)
        advance!(a, 60.0)
        advance!(b, 60.0)
        (tuning_error(a, 1), tuning_error(b, 1), coherence(a), partial_coherence(a, 1))
    end) for s in seeds]
    report_range("обертоны: со строем", [h[1] for h in held])
    report_range("обертоны: без строя", [h[2] for h in held])
    report_range("во сколько раз без строя хуже", [h[2] / h[1] for h in held])
    report_range("целое при узкой полосе", [h[3] for h in held])
    report_range("по совпадению с первой модой", [h[4] for h in held])
    println("   Свойство здесь не «во сколько раз», а «всегда больше»: разброс —")
    println("   это ширина, а не утверждение.")
    return nothing
end

"""
    main()

Прочитать описание: сцена за сценой, без вопросов к природе.

1. Одна нота у каждой вибрации: среда сходится, наложение, толчок, память.
2. Аккорд: у вибрации три частоты; строй держит их в отношении,
   узкая полоса собирает среду не в одно целое, а по совпадениям частот.
3. Разброс: то же чтение при шести начальных условиях — где свойство,
   а где бросок генератора.
"""
function main()
    @printf("%d вибраций, шаг описания %.2f с\n\n", N, DT)

    println("--- одна нота у каждой вибрации " * "-"^28)
    m = build()
    show(m, "сразу")
    advance!(m, 60.0)
    show(m, "через минуту")

    println("\nналожение: что слышно в одной точке среды")
    for t in (0.0, 0.5, 1.0, 2.0)
        @printf("   момент %4.1f с   %+.4f\n", t, wave(m, t))
    end

    println("\nзатухание: толчок и возвращение к своей частоте")
    perturb!(m, 1, 1.5)
    @printf("   после толчка         отклонение %.3f\n", drift(m, 1))
    advance!(m, 6.0)
    @printf("   через шесть секунд   отклонение %.3f\n", drift(m, 1))

    println("\nпамять: связь деформируется тем, что через неё прошло")
    untouched = build()
    routed = build()
    @printf("   связь[1,2] у нетронутой среды      %.6f\n", untouched.coupling[1, 2])
    for _ in 1:200
        perturb!(routed, 1, 1.5)
        advance!(routed, 6.0)
    end
    @printf("   связь[1,2] у слышавшей двести раз  %.6f\n", routed.coupling[1, 2])
    show(untouched, "нетронутая среда")
    show(routed, "среда с руслом")

    println("\nодно и то же событие в двух средах")
    perturb!(untouched, 1, 1.5)
    perturb!(routed, 1, 1.5)
    advance!(untouched, 6.0)
    advance!(routed, 6.0)
    @printf("   нетронутая: отклонение %.3f\n", drift(untouched, 1))
    @printf("   с руслом:   отклонение %.3f\n", drift(routed, 1))
    println("   русло — не запись о событии. Это среда, которая стала другой.")

    println("\n--- аккорд: у вибрации не одна частота, а три " * "-"^15)
    a = build(modes = 3, band = 0.1, tuning = 8.0)
    println("   свои частоты первой вибрации   " *
            join([@sprintf("%.2f", x) for x in a.omega0[1, :]], "  "))
    println("   звук аккорда в четырёх точках  " *
            join([@sprintf("%+.3f", chord(a, 1, t)) for t in (0.0, 0.5, 1.0, 2.0)], "  "))
    advance!(a, 60.0)
    println("   через минуту                   " *
            join([@sprintf("%.2f", x) for x in a.omega[1, :]], "  "))
    show(a, "среда из аккордов")
    @printf("   строй первой вибрации          %.4f\n", tuning_error(a, 1))
    println("   среда не сошлась в одно целое: три голоса живут на своих частотах.")

    println("\nтот же аккорд без строя: половина вибрации уезжает")
    b = build(modes = 3, band = 0.1, tuning = 0.0)
    advance!(b, 60.0)
    println("   через минуту                   " *
            join([@sprintf("%.2f", x) for x in b.omega[1, :]], "  "))
    @printf("   строй первой вибрации          %.4f\n", tuning_error(b, 1))
    println("   обертоны не следуют за основной: аккорд не одно целое.")

    println("\nто же с широкой связью: совпадения не различаются")
    c = build(modes = 3, band = Inf, tuning = 8.0)
    advance!(c, 60.0)
    show(c, "широкая связь")
    @printf("   строй первой вибрации          %.4f\n", tuning_error(c, 1))
    println("   полоса решает, что с чем сходится: узкая различает, широкая — нет.")

    ranges()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

