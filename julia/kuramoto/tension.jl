# Entrain. Курамото: λ — не число среды, а натяжение.
#
# Вопрос поставлен прямо: зависит ли λ от чего-нибудь в модели — от полосы, от
# числа уровней, от силы связи, от шага. Если зависит — λ не настройка, а
# производная величина. Если нет — либо настройка, либо что-то глубже.
# Проверка здесь ровно одна: зависимость. Не «универсальность» — её на одной
# модели не проверить.
#
# λ входит в уравнения одним членом (basic.jl):
#
#     drive[i, p] -= tuning * (omega[i, p] - p * omega[i, 1])    (p > 1),
#
# и соперники у него в модели есть, все названные: связь между вибрациями
# (link/N), связь внутри вибрации (inner — тот же масштаб), затухание γ
# (тянет к своим частотам), полоса Π (что вообще слышно) и шаг описания dt.
# Отсюда мера: промах обертона
#
#     x = |ω_p − p·ω₁| / (p·ω₁)   — в долях его же частоты,
#
# и «аккорд держится» = «промах не хуже написанного» (обертоны в build писались
# с промахом 1 %). Требование одно во всех строках — иначе мерить нечего.
#
# Уравнения те же, что в basic.jl. Шаг здесь задаётся явно: step! берёт dt
# аргументом, поэтому шаг можно менять, не трогая basic.jl.
#
# Итог этого чтения — в журнале, запись в десятый раз: λ — производная, а не
# настройка. В зависимость от записанных параметров он не собирается (ведёт его
# только полоса; сила связи ведёт грубо, число уровней и γ — вовсе нет), зато
# собирается в формулу — равновесие по обертону, — и в ней λ·требование есть тяга
# при равновесии на единицу частоты обертона. И у явного шага есть потолок: он про
# произведение λ·dt, а не про λ и не про dt по отдельности.
#
# Запуск:
#
#     julia julia/kuramoto/tension.jl

include(joinpath(@__DIR__, "basic.jl"))

const NEEDS = (0.10, 0.02, 0.01)   # что считается «держится»: промах в долях
const LADDER = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0,
                64.0, 96.0, 128.0)   # до 128: выше — уже не «нужно», а потолок
const FINE20 = (4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0, 56.0,
                64.0, 80.0, 96.0)    # частая лестница для сбитого аккорда
const SEEDS = 1:6

"""
    x_over(medium)

Промах обертонов: наибольшее из |ω_p − p·ω₁| / (p·ω₁) по всем вибрациям и всем
обертонам. Одна мера на все строки чтения: чем меньше, тем ближе аккорд к
своему отношению. Возвращается и место — (вибрация, обертон), — чтобы было
видно, где именно держит плохо.
"""
function x_over(m::Medium)
    vibs, modes = size(m.omega)
    worst, at = 0.0, (1, 1)
    for i in 1:vibs, p in 2:modes
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        x = abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base)
        if x > worst
            worst, at = x, (i, p)
        end
    end
    return worst, at
end

"""
    drag(medium, i, p)

Тяга связи на моду (i, p): тот же член, что в `step!`, но без строя и без
затухания. Это то, с чем λ спорит; равновесие будет: λ·x + γ·(x − x₀) = D.
"""
function drag(m::Medium, i::Int, p::Int)
    d = 0.0
    for j in 1:size(m.omega, 1)
        j == i && continue
        for n in 1:size(m.omega, 2)
            s = overlap(m.omega[j, n] - m.omega[i, p], m.band)
            d += m.coupling[i, j] * s * sin(m.theta[j, n] - m.theta[i, p])
        end
    end
    return d
end

"""
    advance_step!(medium, seconds, dt)

Прожить средой время при шаге dt. Шаг здесь не `DT`, а аргумент: λ придётся
читать против шага, и это единственный способ, который ничего не ломает в
`basic.jl`. Если перестало быть числом не только состояние, но и фаза — шаг
слишком велик для такой силы, и это тоже чтение, а не сбой. Фаза при этом
сдаётся первой: `omega` ещё конечно, а `theta` уже убежала — потому и проверяются
обе.
"""
function advance_step!(m::Medium, seconds::Float64, dt::Float64)
    for _ in 1:round(Int, seconds / dt)
        try
            step!(m, dt)
        catch
            # Срыв: фаза убежала внутри шага, и след в том же шаге читает уже
            # бесконечную фазу. Это отметка о срыве, а не сбой программы: так
            # видно, где кончается описание при таком шаге.
            fill!(m.omega, Inf)
            fill!(m.theta, Inf)
            return m
        end
        (all(isfinite, m.omega) && all(isfinite, m.theta)) || return m
    end
    return m
end

"""
    tensioned(; seed, modes, band, link, tuning, damping, inner, memory, detune)

Среда того же вида, что в `basic.jl`, с одним добавлением: написанный промах
аккорда можно задать (`detune`) — обертоны сбиваются со своих мест, как их
сбивает нынешняя запись (в `build` промах 1 % и он случайный). При `detune = 0`
ничего не меняется: берётся то, что написал `build`. Остальные записи — те же,
что в `build`, и названы затем, что λ придётся читать против них: против связи,
полосы, затухания, шага — и против следа, который связь наводит сам (`memory`).
"""
function tensioned(; seed::Int = 1, modes::Int = 3, band::Float64 = 0.1,
                    link::Float64 = 0.6, tuning::Float64 = 0.0,
                    damping::Float64 = 0.8, inner::Float64 = 0.0,
                    memory::Float64 = 1e-3, inner_memory::Float64 = 0.0,
                    detune::Float64 = 0.0)
    m = build(modes = modes, band = band, link = link, tuning = tuning,
              damping = damping, inner = inner, memory = memory,
              inner_memory = inner_memory, seed = seed)
    if detune != 0.0
        rng = MersenneTwister(seed + 1000)
        for i in 1:size(m.omega0, 1), p in 2:size(m.omega0, 2)
            m.omega0[i, p] *= 1 + detune * randn(rng)
        end
        m.omega .= m.omega0
    end
    return m
end

"""
    lam_star(need; seconds, dt, ladder, kwargs...)

Наименьшее λ из лестницы, при котором аккорд держится: промах `x ≤ need` у всех
зёрен сразу, через `seconds` секунд. Возвращается λ и промах при нём. «Наименьшее
из лестницы» — честнее, чем корень уравнения: число λ* — не свойство среды, а
ответ на вопрос, и вопросов в чтении несколько. Лестница кончается на 128: если и
там не держится, это и есть ответ, а не повод гнать дальше. Своя, более частая
лестница задаётся, когда λ* проверяется против формулы: тогда важна не ступень, а
то, попадает ли предсказанное число в измеренное.
"""
function lam_star(need::Float64; seconds::Float64 = 60.0, dt::Float64 = DT,
                  ladder = LADDER, kwargs...)
    last = NaN
    for lam in ladder
        worst = 0.0
        for s in SEEDS
            m = tensioned(seed = s, tuning = lam; kwargs...)
            advance_step!(m, seconds, dt)
            worst = max(worst, x_over(m)[1])
        end
        worst <= need && return lam, worst
        last = worst
    end
    return NaN, last
end

"Наибольшая тяга связи на обертонах при данном λ — по тем же шести зёрнам."
function drag_of(; seconds::Float64 = 60.0, dt::Float64 = DT, lam::Float64 = 64.0,
                 kwargs...)
    worst = 0.0
    for s in SEEDS
        m = tensioned(seed = s, tuning = lam; kwargs...)
        advance_step!(m, seconds, dt)
        _, (i, p) = x_over(m)
        worst = max(worst, abs(drag(m, i, p)))
    end
    return worst
end

"""
    held_force(medium, i, p)

То, против чего держит строй: тяга связи на обертон, тяга связи на **основную**
частоту и написанный промах. Из равновесия следует, что промах обертона равен

    |D_p + γ·r₀_p − p·D₁| / ((λ + γ)·p·ω₁),

где D₁ — тяга на основную частоту той же вибрации. Основную λ не держит вовсе:
на неё λ не действует, а связь её тянет наравне со всеми, и её уход входит в
промах обертона множителем p. Без этого слагаемого формула не сходится: она
даёт от тридцатой доли до двух третей промаха — это и видно в чтении.
"""
function held_force(m::Medium, i::Int, p::Int)
    r0 = m.omega0[i, p] - p * m.omega0[i, 1]
    return abs(drag(m, i, p) + m.damping * r0 - p * drag(m, i, 1))
end

"""
    lam_txt(lam)

λ* к печати: `NaN` значит «на лестнице не нашлось» — это тоже ответ.
"""
lam_txt(lam::Float64) = isnan(lam) ? "не нашлось" : @sprintf("%6.1f", lam)

"""
    span_txt(values)

Ряд чисел к печати: наименьшее … наибольшее. Если в ряду сплошь «не нашлось»,
так и говорится — ряд чисел о среде приводится размахом, а не одним числом.
"""
function span_txt(values)
    good = filter(!isnan, values)
    isempty(good) && return "не нашлось"
    return @sprintf("%6.1f … %-6.1f", minimum(good), maximum(good))
end

"""
    lam_seed(need, seed; ladder, kwargs...)

λ*, измеренное по одному зерну: наименьшее λ из лестницы, при котором промах этого
зерна не хуже требования. По зёрнам, а не «наибольшее из шести», — иначе проверка
формулы сравнивала бы числа, снятые с разных зёрен.
"""
function lam_seed(need::Float64, s::Int; seconds::Float64 = 60.0, dt::Float64 = DT,
                  ladder = LADDER, kwargs...)
    for lam in ladder
        m = tensioned(seed = s, tuning = lam; kwargs...)
        advance_step!(m, seconds, dt)
        x_over(m)[1] <= need && return lam
    end
    return NaN
end

"""
    per_second(lam; kwargs...)

Насколько промах сдвинулся за последнюю секунду: ноль — состояние стоит, и
равновесие о нём что-то говорит; много — оно ещё идёт, и формула к нему не
относится.
"""
function per_second(lam::Float64; kwargs...)
    worst = 0.0
    for s in SEEDS
        m = tensioned(seed = s, tuning = lam; kwargs...)
        advance_step!(m, 59.0, DT)
        before = x_over(m)[1]
        advance_step!(m, 1.0, DT)
        worst = max(worst, abs(x_over(m)[1] - before))
    end
    return worst
end

"""
    peak_pred_seed(need, seed; lam, seconds, dt, every, kwargs...)

Предсказание λ* по наибольшей силе за время, а не по снимку: состояние здесь
только кажется стоящим — фазы идут, и сила, против которой держит строй, гуляет,
а держать придётся против самой большой. Из той же формулы:

    λ* = (наибольшая |D_p + γ·r₀_p − p·D₁| за время)/(требование · p·ω₁) − γ.

Сила снимается при λ (там, где строй уже держит). Ничего не подгоняется: γ и r₀ —
из записи, требование — из вопроса, наибольшая сила — из прогона.
"""
function peak_pred_seed(need::Float64, s::Int; lam::Float64 = 16.0,
                        seconds::Float64 = 60.0, dt::Float64 = DT,
                        every::Float64 = 0.5, kwargs...)
    m = tensioned(seed = s, tuning = lam; kwargs...)
    vibs, modes = size(m.omega)
    peaks = zeros(vibs, modes)
    for n in 1:round(Int, seconds / dt)
        step!(m, dt)
        (all(isfinite, m.omega) && all(isfinite, m.theta)) || break
        n % round(Int, every / dt) == 0 || continue
        for i in 1:vibs, p in 2:modes
            peaks[i, p] = max(peaks[i, p], held_force(m, i, p))
        end
    end
    worst = 0.0
    for i in 1:vibs, p in 2:modes
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        worst = max(worst, peaks[i, p] / (need * p * base) - m.damping)
    end
    return worst
end

"""
    grows(lam, dt; seconds, push, kwargs...)

Во сколько раз вырос промах за `seconds` секунд после толчка обертону. Больше
единицы — толчок растёт, меньше — гаснет. Это про потолок силы при данном шаге и
без порогов: смотрится знак — растёт или гаснет, а не величина.
"""
function grows(lam::Float64, dt::Float64; seconds::Float64 = 1.0,
               push::Float64 = 0.05, seeds = 1:2, kwargs...)
    worst = 0.0
    for s in seeds
        m = tensioned(seed = s, tuning = lam; kwargs...)
        advance_step!(m, 5.0, dt)
        perturb!(m, 1, push, 2)
        before = x_over(m)[1]
        advance_step!(m, seconds, dt)
        # Сорвавшееся состояние — не «гаснет»: при таком шаге описание кончилось.
        (all(isfinite, m.omega) && all(isfinite, m.theta)) || return NaN
        worst = max(worst, x_over(m)[1] / max(before, 1e-300))
    end
    return worst
end

"""
    grows_txt(ratio)

Рост к печати: сорвавшееся состояние — это «не сосчитать», а не «гаснет».
"""
grows_txt(ratio::Float64) = isfinite(ratio) ? @sprintf("×%-9.2e", ratio) : "сорвалась "

"""
    permute_modes!(medium, σ)

Переставить индексы мод: столбцы `omega0`, `omega` и `theta` — и связь внутри
вибрации той же перестановкой. У миров свои `omega`, `omega0`, `theta` — это
взгляды на те же строки, поэтому переставлять их не нужно; а связь своя, её
переставляют.
"""
function permute_modes!(m::Medium, σ::Vector{Int})
    m.omega0 .= m.omega0[:, σ]
    m.omega .= m.omega[:, σ]
    m.theta .= m.theta[:, σ]
    if m.inside !== nothing
        for w in m.inside
            w.coupling .= w.coupling[σ, σ]
        end
    end
    return m
end

"""
    unpermuted(medium, σ)

То же состояние, но моды возвращены туда, откуда их сдвинула перестановка: после
подмены новая мода j держит старую σ[j], значит обратно — столбцы σ⁻¹. Нужно
затем, что сверять переставленное с непереставленным можно только по одним и тем
же модам: иначе сравнивались бы разные аккорды, и разница нашлась бы всегда. Миров
у этого состояния нет: здесь читаются одни числа, а не среды.
"""
function unpermuted(m::Medium, σ::Vector{Int})
    back = invperm(σ)
    return Medium(omega0 = m.omega0[:, back], omega = m.omega[:, back],
                  theta = m.theta[:, back], coupling = copy(m.coupling), band = m.band,
                  tuning = m.tuning, memory = m.memory, damping = m.damping)
end

"""
    main()

Прочитать зависимость: держится ли λ на чём-нибудь, кроме себя.

1. Шаг проверки: сколько нужно — промах без строя и λ* при трёх требованиях.
2. От чего зависит λ*: сила связи, число уровней, полоса, затухание, промах.
3. Формула вместо таблицы: равновесие λ·x + γ·(x − x₀) = D и предсказание λ*.
4. Потолок: λ против шага описания.
5. Основная частота: перестановка индексов мод при λ = 0 и при λ > 0.
"""
function main()
    @printf("λ: у вибрации 3 моды, полоса 0.10, связь 0.60, затухание 0.8, шаг %.3f с\n\n",
            DT)

    println("--- шаг проверки: сколько нужно, чтобы держалось " * "-"^8)
    println("   Аккорд 1 : 2 : 3 с написанным промахом 1 %. «Держится» — промах не")
    println("   хуже написанного, и требование это одно на всё чтение.")
    for need in NEEDS
        lam, worst = lam_star(need)
        @printf("   требование %4.0f %%   λ* %s   промах при нём %.6f   λ*·требование %7.4f\n",
                100 * need, lam_txt(lam), worst, lam * need)
    end
    x0 = 0.0
    for s in SEEDS
        m0 = tensioned(seed = s, tuning = 0.0)
        advance_step!(m0, 60.0, DT)
        x0 = max(x0, x_over(m0)[1])
    end
    @printf("   без строя (λ = 0) промах %.6f   тяга связи при λ = 64   %.4f\n",
            x0, drag_of(lam = 64.0))
    println("   «Не нашлось» в столбце λ* значит, что и на 128 не держится, — это")
    println("   тоже ответ. Промах без строя — это написанный, растрёпанный связью:")
    println("   тяга его не улучшает. Требование двигает ответ, но не подчиняет его")
    println("   целиком: у строгих требований (2 % и 1 %) произведение λ*·требование")
    println("   стоит на одном числе — оно и есть сила, против которой держат: тяга")
    println("   при равновесии на единицу частоты обертона (в строке выше — того же")
    println("   порядка). А при 10 % оно ниже: держать почти нечего,")
    println("   и λ* упирается не в закон, а в дно.")

    println("\n--- от чего зависит λ* (требование 1 %) " * "-"^8)
    println("   Сила связи между вибрациями: та же тяга, тот же спор. Строка без следа —")
    println("   контроль: при нулевой силе связь наводит только след.")
    for (link, mem) in ((0.0, 1e-3), (0.0, 0.0), (0.3, 1e-3), (0.6, 1e-3),
                        (1.2, 1e-3), (2.4, 1e-3))
        lam, worst = lam_star(0.01, link = link, memory = mem)
        dd = drag_of(link = link, memory = mem)
        @printf("   сила %4.2f  след %s   λ* %s   промах %.6f   тяга %8.5f\n",
                link, mem == 0.0 ? "нет " : "есть", lam_txt(lam), worst, dd)
    end
    println("   При нулевой силе λ* не ноль: связь наводит сам след — того, что через")
    println("   неё прошло. И видно прямо: тяга не ноль при силе 0.00. Контроль без")
    println("   следа даёт тягу 0.00000 и λ* 2.0 — дно лестницы: держать там нечего,")
    println("   кроме написанного промаха, и λ уходит на него.")
    println("   Число уровней: сколько обертонов держать.")
    for modes in (2, 3, 5, 8)
        lam, worst = lam_star(0.01, modes = modes)
        @printf("   уровней %d   λ* %s   промах %.6f   тяга %8.5f\n",
                modes, lam_txt(lam), worst, drag_of(modes = modes))
    end
    println("   Полоса отклика: что вообще слышно.")
    for band in (0.02, 0.05, 0.1, 0.5, Inf)
        lam, worst = lam_star(0.01, band = band)
        @printf("   полоса %5.2f   λ* %s   промах %.6f   тяга %8.5f\n",
                band, lam_txt(lam), worst, drag_of(band = band))
    end
    println("   Затухание: γ тянет к написанному — и мешает, и помогает.")
    for damp in (0.0, 0.4, 0.8, 1.6, 3.2)
        lam, worst = lam_star(0.01, damping = damp)
        @printf("   γ %4.1f   λ* %s   промах %.6f   λ*/γ %8.3f\n",
                damp, lam_txt(lam), worst, damp == 0.0 ? NaN : lam / damp)
    end
    println("   Написанный промах: чем хуже написан аккорд, тем больше промах при том")
    println("   же λ. Хуже того: легче тяге от этого не делается — λ* если и")
    println("   двигается, то вверх (на другом потоке случайных чисел он уходит на")
    println("   ступень выше), а промах при нём подбирается к самому требованию:")
    println("   сбитость входит в равновесие тем же весом, что и тяга.")
    for det in (0.0, 0.01, 0.05, 0.2)
        lam, worst = lam_star(0.01, detune = det)
        @printf("   промах %5.3f   λ* %s   промах при нём %.6f\n",
                det, lam_txt(lam), worst)
    end
    println("   Связь внутри вибрации (сила снаружи убрана, чтобы слышно было её одну).")
    for inner in (0.0, 0.2, 0.6, 1.2)
        lam, worst = lam_star(0.01, link = 0.0, inner = inner)
        @printf("   внутри %4.2f   λ* %s   промах %.6f   тяга %8.5f\n",
                inner, lam_txt(lam), worst, drag_of(link = 0.0, inner = inner))
    end
    println("   Ведёт λ* одно: полоса. Чем уже слышно, тем меньше тяга и тем ниже λ*;")
    println("   чем шире — тем выше. Это и единственная строка, которая идёт порядком")
    println("   при смене потока случайных чисел, — а поток у 1.10 и 1.13 разный.")
    println("   Сила связи ведёт грубо: от 0.3 к 0.6 λ* растёт, а дальше порядок")
    println("   рвётся — при самой сильной связи λ* может оказаться и ниже, чем при")
    println("   слабой: тяга снимается на том состоянии, которое само от λ и зависит.")
    println("   Число уровней и γ не ведут вовсе: разброс в пределах ступени лестницы,")
    println("   а местами и до дна. Оттого λ* и не собирается в зависимость от")
    println("   записанных параметров. А собирается в неё произведение λ*·требование —")
    println("   оно и есть тяга при равновесии на единицу частоты обертона, — об этом")
    println("   формула ниже.")

    println("\n--- формула, а не таблица " * "-"^16)
    println("   Равновесие по обертону: λ·x + γ·(x − x₀) = D. Но у основной частоты")
    println("   своего строя нет — на неё λ не действует, а связь её тянет как всех,")
    println("   поэтому основной промах равен просто D₁/γ, и он входит в промах")
    println("   обертона множителем p:")
    println("      x_p = |D_p + γ·r₀_p − p·D₁| / ((λ + γ)·p·ω₁).")
    println("   Обе версии — со слагаемым и без — против прогона:")
    for lam in (8.0, 16.0, 32.0, 64.0)
        worst_m, worst_n, worst_p = 0.0, 0.0, 0.0
        for s in SEEDS
            m = tensioned(seed = s, tuning = lam)
            advance_step!(m, 60.0, DT)
            xm, (i, p) = x_over(m)
            xm > worst_m || continue
            base = abs(m.omega[i, 1])
            r0 = m.omega0[i, p] - p * m.omega0[i, 1]
            worst_m = xm
            worst_p = held_force(m, i, p) / ((lam + m.damping) * p * base)
            worst_n = abs(drag(m, i, p) + m.damping * r0) / ((lam + m.damping) * p * base)
        end
        @printf("   λ %5.1f   измерен %.6f   формула %.6f   без основной %.6f   ход %.1e\n",
                lam, worst_m, worst_p, worst_n, per_second(lam))
    end
    println("   Промах — это стоящее число, а сила, против которой держат, гуляет: фазы")
    println("   идут, и основная частота уезжает вместе со всеми. Поэтому λ* считается")
    println("   по наибольшей силе за минуту, а не по снимку. Проверка по зёрнам:")
    println("   у каждого зерна своё предсказание и своё измерение; к печати — размах")
    println("   по зёрнам, лестница частая.")
    for (need, ladder) in ((0.10, (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)),
                           (0.02, (8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0)),
                           (0.01, (16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 48.0, 56.0)))
        preds = [peak_pred_seed(need, s; lam = 16.0) for s in SEEDS]
        preds2 = [peak_pred_seed(need, s; lam = 48.0) for s in SEEDS]
        meas = [lam_seed(need, s; ladder = ladder) for s in SEEDS]
        @printf("   требование %4.0f %%   по наибольшей силе %s   (сила снята при 48: %s)\n",
                100 * need, span_txt(preds), span_txt(preds2))
        @printf("   %19s измерено %s\n", "", span_txt(meas))
    end
    println("   Предсказание идёт впереди измерения: разбросы перекрываются, порядок")
    println("   один, но сила снята при одном λ, а нужна при том, который только")
    println("   ищется, — оттого предсказание и стоит выше: где на четверть, где почти")
    println("   вдвое. На месте же, где λ задан и равновесие стоит, совпадение до")
    println("   третьего знака: таблица выше. А без слагаемого с тягой на основную")
    println("   частоту формула не сходится: она даёт от тридцатой доли до двух третей")
    println("   промаха, и в строке «без основной» это видно.")
    println("   И проверка на новом месте: у сбитого на 20 % аккорда γ входит в")
    println("   равновесие своим весом, и формула обещает, что λ* пойдёт за γ.")
    println("   Здесь она уже не описывает, а говорит вперёд.")
    for damp in (0.4, 0.8, 3.2)
        preds = [peak_pred_seed(0.01, s; lam = 16.0, damping = damp, detune = 0.2)
                 for s in SEEDS]
        meas = [lam_seed(0.01, s; ladder = FINE20, damping = damp, detune = 0.2)
                for s in SEEDS]
        @printf("   γ %4.1f   по формуле %s   измерено %s\n",
                damp, span_txt(preds), span_txt(meas))
    end

    println("\n--- потолок: λ против шага описания " * "-"^6)
    println("   Здесь не «до бесконечности», а знак: гаснет толчок или растёт. Проба")
    println("   идёт прямо в единицах λ·шаг — предел у явного шага про произведение.")
    for dt in (0.005, 0.01, 0.04)
        @printf("   шаг %6.4f ", dt)
        for k in (0.5, 1.0, 1.5, 2.0, 3.0)
            @printf("  λ·шаг %3.1f: %s", k, grows_txt(grows(k / dt, dt)))
        end
        println()
    end
    println("   Знак меняется между λ·шаг 1.5 и 2.0 — при всех трёх шагах. У самого")
    println("   предела числа совпадают до второго знака (×2.2 при любом шаге):")
    println("   гаснет и растёт произведение, а не сила и не шаг. Ниже предела числа за")
    println("   секунду разные и зависят от шага (при λ·шаг 0.5 гашение на грубом шаге")
    println("   слабее в разы): там гасит не только строй — своя скорость есть и у")
    println("   связи. «Сорвалась» в таблице значит, что фаза ушла внутри шага и")
    println("   читать уже нечего, — это не «гаснет».")
    println("   А значение λ при равновесии от шага не зависит вовсе — совпадение до")
    println("   последней печатаемой цифры: держится равновесие, а равновесие о шаге")
    println("   не знает. Потолок ниже нужного λ значит одно: при таком шаге аккорд")
    println("   держать нечем — и это о записи, а не о среде.")
    for dt in (0.0025, 0.005, 0.01, 0.02)
        m = tensioned(seed = 1, tuning = 32.0)
        advance_step!(m, 60.0, dt)
        m2 = tensioned(seed = 1, tuning = 32.0)
        advance_step!(m2, 60.0, DT)
        @printf("   шаг %6.4f   промах при λ = 32   %.6f   (при шаге %.3f: %.6f)\n",
                dt, x_over(m)[1], DT, x_over(m2)[1])
    end

    println("\n--- основная частота: не особенность, а то, что делает λ " * "-"^2)
    println("   Перестановка индексов мод (1 2 3 → 2 3 1): при λ = 0 среда не должна")
    println("   заметить подмены — ничто в законе моду не выделяет. Сверка идёт по одним")
    println("   и тем же модам: переставленное возвращается на место.")
    σ = [2, 3, 1]
    for lam in (0.0, 8.0)
        a = tensioned(seed = 1, tuning = lam)
        b = tensioned(seed = 1, tuning = lam)
        permute_modes!(b, σ)
        step!(a, DT)
        step!(b, DT)
        after_step = maximum(abs.(unpermuted(b, σ).omega .- a.omega))
        advance_step!(a, 59.0, DT)
        advance_step!(b, 59.0, DT)
        back = unpermuted(b, σ)
        after_minute = maximum(abs.(back.omega .- a.omega))
        @printf("   λ %4.1f   за шаг %.3e   за минуту %.3e   промах %.6f против %.6f\n",
                lam, after_step, after_minute, x_over(a)[1], x_over(back)[1])
    end
    println("   При λ = 0 подмены не видно: за шаг — ровно ноль, за минуту — либо тот")
    println("   же ноль, либо последний бит (порядок сложения, а не место). Закон не")
    println("   говорит, какая мода первая. При λ > 0 разница настоящая, и это не")
    println("   изъян: λ и есть то, что делает первую моду первой. Требование")
    println("   «перестановка индексов не должна менять числа» — это требование λ = 0,")
    println("   а λ = 0 аккорд не держит (лестница выше). Пункт закрыт не потому, что")
    println("   стал красивым, а потому, что у него два исхода и оба измерены.")

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
