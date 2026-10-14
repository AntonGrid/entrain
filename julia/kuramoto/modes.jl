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
# За ним второй вопрос, из того же строя: может ли собранное держаться следом.
# Память записывается двумя способами — угол кратной фазы и сама работа уровня;
# разница между ними и есть ответ.
#
# За ним третий: важно ли, что именно помнить. Отбор уровня везде один, а
# запись — разная: работа, срабатывание, отрыв, порог, работа в фазе, меткость.
# Различие между ними оказалось не в содержании, а в разборчивости записи —
# и у разборчивости есть цена.
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
    step_trace!(medium, dt, strength, C, link, eps, work)

То же, что `step_levels!`, но запись следа вынесена наружу вопросом: что
именно помнится. `work = false` — след о фазе (`cos(k·θ_p − j·θ_q)`), как
раньше: помнит угол кратной фазы. `work = true` — след о работе уровня
(`S(k·ω_p − j·ω_q)`): помнит, что уровень был близок, то есть что связь
через него шла.

Разница не в величине, а в знаке: фазовый след у разных вибраций разного
знака и при усреднении гасится; след о работе неотрицателен и копится там,
где уровень стоял.
"""
function step_trace!(m::Medium, dt::Float64, strength::Float64,
                     C::Matrix{Float64}, link::Matrix{Float64},
                     eps::Float64, work::Bool)
    vibs, modes = size(m.omega)
    K = size(C, 1)
    drive = zeros(vibs, modes)
    for i in 1:vibs, p in 1:modes, q in 1:modes
        q == p && continue
        for k in 1:K, j in 1:K
            w = strength * C[k, j] * link[k, j]
            w == 0.0 && continue
            s = overlap(k * m.omega[i, p] - j * m.omega[i, q], m.band)
            drive[i, p] += w * s * (j * m.omega[i, q] / k - m.omega[i, p])
        end
    end
    m.omega .+= dt .* (drive .- m.damping .* (m.omega .- m.omega0))
    m.theta .+= dt .* m.omega
    if eps != 0.0
        for k in 1:K, j in 1:K
            acc, n = 0.0, 0
            for i in 1:vibs, p in 1:modes, q in 1:modes
                q == p && continue
                acc += work ? overlap(k * m.omega[i, p] - j * m.omega[i, q], m.band) :
                              cos(k * m.theta[i, p] - j * m.theta[i, q])
                n += 1
            end
            n == 0 && continue
            link[k, j] += dt * eps * (acc / n - link[k, j])
        end
    end
    return m
end

"""
    advance_trace!(medium, seconds, strength, C, link, eps = 0.0, work = true)

Прожить средой указанное время по шагу, где запись следа выбрана вопросом.
"""
function advance_trace!(m::Medium, seconds::Float64, strength::Float64,
                        C::Matrix{Float64}, link::Matrix{Float64},
                        eps::Float64 = 0.0, work::Bool = true)
    for _ in 1:round(Int, seconds / DT)
        step_trace!(m, DT, strength, C, link, eps, work)
    end
    return m
end

"""
    record_signal(S, d, kind)

Что именно помнится об отборе. Отбор везде один: уровень (k, j) — это
отношение k·ω_p ≈ j·ω_q, и работает он тем сильнее, чем точнее совпадение
(матрица `S`), а промах меряется расстройкой `d = k·ω_p − j·ω_q`.

Записывается отбор по-разному:

- `:work`   — сама работа уровня: `S[k, j]`;
- `:winner` — работа того уровня, что за эту пару сработал сильнее всех;
- `:margin` — насколько он обошёл следующего за ним: отрыв;
- `:hold`   — сработал ли он больше половины: порог вместо меры;
- `:sharp`  — точность попадания: `S[k, j] / |d|` — меткость.

Все они берутся по одной паре (i, p, q); сложение идёт после, поэтому
нелинейность (победитель, порог) считается там, где она и происходит — в
паре, а не в среднем. Фазовая работа сюда не входит: она спрашивает не о
паре, а о согласии фаз, и берётся сразу по всем парам (см. `:coherent`).
"""
function record_signal(S::Matrix{Float64}, d::Matrix{Float64}, kind::Symbol)
    K = size(S, 1)
    raw = zeros(K, K)
    if kind === :work
        return copy(S)
    elseif kind === :sharp
        for k in 1:K, j in 1:K
            raw[k, j] = S[k, j] / (abs(d[k, j]) + 1e-3)
        end
        return raw
    end
    best, second, bk, bj = -Inf, -Inf, 1, 1
    for k in 1:K, j in 1:K
        if S[k, j] > best
            second = best
            best, bk, bj = S[k, j], k, j
        elseif S[k, j] > second
            second = S[k, j]
        end
    end
    gap = best - (second == -Inf ? 0.0 : second)
    if kind === :winner
        raw[bk, bj] = best
    elseif kind === :margin
        raw[bk, bj] = gap
    elseif kind === :hold
        for k in 1:K, j in 1:K
            raw[k, j] = S[k, j] > 0.5 ? 1.0 : 0.0
        end
    else
        error("kind: $kind")
    end
    return raw
end

"""
    step_record!(medium, dt, strength, C, link, eps, kind)

Тот же шаг, что `step_trace!`, но запись памяти выбрана вопросом «что помнить».

Отбор уровня — один и тот же во всех случаях (ближайший уровень тянет частоту).
Меняется только то, чем отбор записан: `record_signal` берёт его по каждой
паре, а `:coherent` — по всем сразу:

- `:coherent` — работа, взятая в фазе: |⟨S·e^(i·Δθ)⟩|, то есть работа, что была
                согласна, а не просто шла.

Сравнение честное только при общей мере: запись приводится к среднему 1.
"""
function step_record!(m::Medium, dt::Float64, strength::Float64,
                      C::Matrix{Float64}, link::Matrix{Float64},
                      eps::Float64, kind::Symbol)
    vibs, modes = size(m.omega)
    K = size(C, 1)
    drive = zeros(vibs, modes)
    for i in 1:vibs, p in 1:modes, q in 1:modes
        q == p && continue
        for k in 1:K, j in 1:K
            w = strength * C[k, j] * link[k, j]
            w == 0.0 && continue
            s = overlap(k * m.omega[i, p] - j * m.omega[i, q], m.band)
            drive[i, p] += w * s * (j * m.omega[i, q] / k - m.omega[i, p])
        end
    end
    m.omega .+= dt .* (drive .- m.damping .* (m.omega .- m.omega0))
    m.theta .+= dt .* m.omega
    eps == 0.0 && return m

    if kind === :coherent
        re = zeros(K, K)
        im = zeros(K, K)
        n = 0
        for i in 1:vibs, p in 1:modes, q in 1:modes
            q == p && continue
            n += 1
            for k in 1:K, j in 1:K
                s = overlap(k * m.omega[i, p] - j * m.omega[i, q], m.band)
                d = k * m.theta[i, p] - j * m.theta[i, q]
                re[k, j] += s * cos(d)
                im[k, j] += s * sin(d)
            end
        end
        n == 0 && return m
        for k in 1:K, j in 1:K
            raw = sqrt(re[k, j]^2 + im[k, j]^2) / n
            link[k, j] += dt * eps * (raw - link[k, j])
        end
        return m
    end

    acc = zeros(K, K)
    n = 0
    for i in 1:vibs, p in 1:modes, q in 1:modes
        q == p && continue
        n += 1
        S = zeros(K, K)
        d = zeros(K, K)
        for k in 1:K, j in 1:K
            S[k, j] = overlap(k * m.omega[i, p] - j * m.omega[i, q], m.band)
            d[k, j] = k * m.omega[i, p] - j * m.omega[i, q]
        end
        acc .+= record_signal(S, d, kind)
    end
    n == 0 && return m
    raw = acc ./ n
    for k in 1:K, j in 1:K
        link[k, j] += dt * eps * (raw[k, j] - link[k, j])
    end
    return m
end

"""
    advance_record!(medium, seconds, strength, C, link, eps, kind)

Прожить средой время по шагу, где запись памяти выбрана вопросом.
"""
function advance_record!(m::Medium, seconds::Float64, strength::Float64,
                         C::Matrix{Float64}, link::Matrix{Float64},
                         eps::Float64, kind::Symbol)
    for _ in 1:round(Int, seconds / DT)
        step_record!(m, DT, strength, C, link, eps, kind)
    end
    return m
end

"Запись, приведённая к среднему 1: сравниваются различение, а не сила."
function to_carrier(link::Matrix{Float64})
    ml = mean(link)
    return ml == 0.0 ? copy(link) : abs.(link ./ ml)
end

"Доля веса на первом по величине уровне: насколько запись говорит о прочем."
function top_share(link::Matrix{Float64})
    s = sum(abs.(link))
    return s == 0.0 ? 0.0 : maximum(abs.(link)) / s
end


"""
    level_bias(K, level, bias)

Та же шкала, но одному уровню отдан больший вес: среда с такой шкалой
собирает не то, что собирают все, — на нём и видно, несёт ли след выбор.
"""
function level_bias(K::Int, level::Tuple{Int, Int}, bias::Float64)
    C = level_weights(K)
    for k in 1:K, j in 1:K
        if (k, j) == level || (j, k) == level
            C[k, j] *= bias
        end
    end
    return C
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
    println("   Толчок обертону возвращают оба, но не под всяким начальным условием:")
    println("   и у λ = 2.0, и у отклика с узкой полосой остаётся иногда больше самого")
    println("   толчка. Разница в другом — и видна она на толчке основной.")
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
    println("   привязка есть: на 1.10 отношение он держит строже λ, на 1.13 —")
    println("   наравне, а при узкой полосе 0.10 бывает и хуже λ.")

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

    println("\n   а важно ли, что веса уровней разные: та же связь, вес ровный.")
    odd = [(begin
        m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
        detune!(m, 0.08, 900 + s)
        advance_levels!(m, 60.0, 8.0, level_weights(4), ones(4, 4))
        (found_inside(m, 4, 0.02)[1], mean(unison(m, i) for i in 1:N))
    end) for s in SEEDS]
    even = [(begin
        m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
        detune!(m, 0.08, 900 + s)
        advance_levels!(m, 60.0, 8.0, ones(4, 4), ones(4, 4))
        (found_inside(m, 4, 0.02)[1], mean(unison(m, i) for i in 1:N))
    end) for s in SEEDS]
    @printf("   вес 1/(k·j)   нашлось %2d из %2d   сжатие %.3f … %.3f\n",
            sum(o[1] for o in odd), 3 * N * length(SEEDS),
            minimum(o[2] for o in odd), maximum(o[2] for o in odd))
    @printf("   вес ровный    нашлось %2d из %2d   сжатие %.3f … %.3f\n",
            sum(e[1] for e in even), 3 * N * length(SEEDS),
            minimum(e[2] for e in even), maximum(e[2] for e in even))
    println("   Ровный вес собирает не меньше, а сжатие при нём того же порядка, что при")
    println("   весе 1/(k·j): иерархия весов не нужна — шкала это разрешение, а не порядок.")

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
    println("   аккорд и держался, только взятые выше: (2,4) на 1.10 — то же 1:2, (1,3)")
    println("   на 1.13 — терция. Среда закрепляет не событие, а то, что сходилось.")

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

    println("\n--- носитель: может ли след держать строй " * "-"^14)
    println("   собрать аккорд, дать среде навести след, затем ослабить силу и")
    println("   спросить, что осталось. Следа две записи: помнить угол кратной фазы")
    println("   (как выше) или помнить саму работу уровня. Память 0.5, сборка 240 с.")
    K = 4
    C = level_weights(K)
    flat = ones(K, K)
    sled = Dict{Bool, Matrix{Float64}}()
    for work in (false, true)
        m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = 1)
        detune!(m, 0.08, 901)
        l = ones(K, K)
        advance_trace!(m, 240.0, 16.0, C, l, 0.5, work)
        sled[work] = l
        close, pairs, _ = found_inside(m, K, 0.02)
        @printf("   след о %-6s собрано %2d из %2d   %s\n",
                work ? "работе" : "фазе", close, pairs, link_top(l, 4))
        @printf("   %14s разброс %.3f … %.3f\n", "", minimum(l), maximum(l))
    end
    println("   Фазовый след у разных вибраций разного знака и при усреднении гасится;")
    println("   след о работе неотрицателен и копится там, где уровень стоял: одна")
    println("   запись помнит угол, другая — что связь через уровень шла.")

    println("\n   тот же след в чужой среде, где заданной шкалы нет вовсе; вес следа")
    println("   приведён к среднему 1, чтобы сравнивать не силу, а различение.")
    for (name, carrier) in (("след о работе", sled[true] ./ mean(sled[true])),
                            ("след о фазе", sled[false] ./ mean(sled[false])),
                            ("ровный", flat))
        for strength in (1.0, 4.0)
            close, uni, err = 0, 0.0, 0.0
            total = Dict{Tuple{Int, Int}, Int}()
            for s in SEEDS
                m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
                detune!(m, 0.08, 900 + s)
                advance_trace!(m, 60.0, strength, flat, copy(carrier), 0.0, true)
                c, _, counts = found_inside(m, K, 0.02)
                close += c
                uni += mean(unison(m, i) for i in 1:N)
                err += tuning_error(m, 1)
                for (key, n) in counts
                    total[key] = get(total, key, 0) + n
                end
            end
            n = length(SEEDS)
            @printf("   связь %4.1f   %-13s нашлось %3d из %3d   сжатие %.3f   строй %.4f   %s\n",
                    strength, name, close, 3 * N * n, uni / n, err / n, found_line(total))
        end
    end
    println("   След о работе собирает не меньше всех и держит строй крепче ровного;")
    println("   фазовый след при сильной связи уводит аккорд в совпадение, и строй при")
    println("   нём хуже, чем вовсе без следа. Носителем может быть след, но только")
    println("   тот, что помнит работу связи, а не угол.")

    println("\n   несёт ли след выбор: среда со смещённой шкалой собирает 1:3.")
    m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = 1)
    detune!(m, 0.08, 901)
    l = ones(K, K)
    advance_trace!(m, 240.0, 16.0, level_bias(K, (1, 3), 16.0), l, 0.5, true)
    biased = l ./ mean(l)
    @printf("   след, наведённый на 1:3: %s\n", link_top(biased, 4))
    for (name, carrier) in (("след 1:2", sled[true] ./ mean(sled[true])),
                            ("след 1:3", biased), ("ровный", flat))
        close, err = 0, 0.0
        total = Dict{Tuple{Int, Int}, Int}()
        for s in SEEDS
            m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
            detune!(m, 0.08, 900 + s)
            advance_trace!(m, 60.0, 4.0, flat, copy(carrier), 0.0, true)
            c, _, counts = found_inside(m, K, 0.02)
            close += c
            err += tuning_error(m, 1)
            for (key, n) in counts
                total[key] = get(total, key, 0) + n
            end
        end
        @printf("   %-9s нашлось %3d из %3d   строй %.4f   %s\n",
                name, close, 3 * N * length(SEEDS), err / length(SEEDS), found_line(total))
    end
    println("   Выбор в следе есть: под следом, наведённым на 1:3, вперёд выходит 1:3,")
    println("   под следом на 1:2 — 1:2. След не велит и не задаёт шкалу: он даёт")
    println("   перевес тому, что среда уже держала.")

    println("\n   сила убрана совсем: держит ли что-нибудь аккорд.")
    raw, was, after = 0, 0, 0
    for s in SEEDS
        m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
        detune!(m, 0.08, 900 + s)
        raw += found_inside(m, K, 0.02)[1]
        l = ones(K, K)
        advance_trace!(m, 240.0, 16.0, C, l, 0.5, true)
        was += found_inside(m, K, 0.02)[1]
        advance_trace!(m, 60.0, 0.0, C, l, 0.0, true)
        after += found_inside(m, K, 0.02)[1]
    end
    @printf("   сбитое состояние само по себе %2d из %2d, при сборке %2d из %2d, после уборки силы %2d из %2d\n",
            raw, 3 * N * length(SEEDS), was, 3 * N * length(SEEDS), after, 3 * N * length(SEEDS))
    println("   После уборки — ровно столько, сколько даёт сбитое состояние само по")
    println("   себе: без силы носителя нет и у следа. След — носитель выбора, сила —")
    println("   носитель величины; строй держится обоими сразу, как и всё в описании.")

    println("\n--- а важно ли, что именно помнить: шесть записей одной связи и угол фазы " * "-"^2)
    println("   отбор везде один (ближайший уровень тянет частоту) — меняется только то,")
    println("   чем он записан. Запись приводится к среднему 1, чтобы сравнивать не силу,")
    println("   а различение. Две среды наведения, шесть чужих сред на перенос; строй")
    println("   берётся по всем вибрациям и печатается серединой и худшим — средний")
    println("   обманывает: одна среда, потерявшая аккорд, тянет его на себя.")
    записи = ((:work, "работа"), (:winner, "срабатывание"), (:margin, "отрыв"),
              (:hold, "порог"), (:coherent, "работа в фазе"), (:sharp, "меткость"),
              (:phase, "угол фазы"))
    for (kind, name) in записи
        close, errs = 0, Float64[]
        total = Dict{Tuple{Int, Int}, Int}()
        shares = Float64[]
        for seed in (1, 2)
            m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = seed)
            detune!(m, 0.08, 901)
            l = ones(K, K)
            if kind === :phase
                advance_trace!(m, 240.0, 16.0, C, l, 0.5, false)
            else
                advance_record!(m, 240.0, 16.0, C, l, 0.5, kind)
            end
            push!(shares, top_share(l))
            carrier = to_carrier(l)
            for s in SEEDS
                m2 = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
                detune!(m2, 0.08, 900 + s)
                advance_trace!(m2, 60.0, 4.0, flat, copy(carrier), 0.0, true)
                c, _, counts = found_inside(m2, K, 0.02)
                close += c
                push!(errs, mean(tuning_error(m2, i) for i in 1:N))
                for (key, n) in counts
                    total[key] = get(total, key, 0) + n
                end
            end
        end
        sort!(errs)
        @printf("   %-14s нашлось %3d из %3d   строй %.4f … %.4f   доля первого %.3f   %s\n",
                name, close, 3 * N * length(SEEDS) * 2, errs[cld(length(errs), 2)], errs[end],
                mean(shares), found_line(total))
    end
    for (name, carrier) in (("ровный", flat),)
        close, errs = 0, Float64[]
        total = Dict{Tuple{Int, Int}, Int}()
        for s in SEEDS
            m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
            detune!(m, 0.08, 900 + s)
            advance_trace!(m, 60.0, 4.0, flat, copy(carrier), 0.0, true)
            c, _, counts = found_inside(m, K, 0.02)
            close += c
            push!(errs, mean(tuning_error(m, i) for i in 1:N))
            for (key, n) in counts
                total[key] = get(total, key, 0) + n
            end
        end
        sort!(errs)
        @printf("   %-14s нашлось %3d из %3d   строй %.4f … %.4f   %s\n",
                name, close, 3 * N * length(SEEDS), errs[cld(length(errs), 2)], errs[end],
                found_line(total))
    end
    println("   Работа, срабатывание, порог и отрыв держат строй крепче ровного — по")
    println("   середине и по худшему, и на 1.10, и на 1.13. Работа в фазе крепче по")
    println("   середине, но на 1.13 её худший строй хуже ровного: одной среде аккорд не")
    println("   удержать. Угол фазы серединой рядом с ровным, а худшим много хуже — и на")
    println("   1.13 он уводит аккорд в совпадение, как меткость на 1.10. Меткость")
    println("   держится по-разному: на 1.10 она уводит аккорд в совпадение (в найденном")
    println("   есть 1:1 и 3:3) — тогда точное попадание и есть слияние; на 1.13 этих")
    println("   следов почти нет. Доля первого уровня — мера разборчивости записи, а не")
    println("   крепости строя: она говорит, сколько записано тому, чего не было, и ничего")
    println("   не говорит о том, удержится ли строй.")

    println("\n   и у различения есть цена: несёт ли такая запись выбор.")
    println("   проба одна и та же дважды, меняется лишь смещение шкалы: след наводится")
    println("   ею и переносится в ровную среду. Печатается и то, что записано в")
    println("   носителе, и то, что из этого переносится; у ровного носителя та же доля —")
    println("   мерило: столько среда даёт сама, и выбирать в нём нечего. В скобках у")
    println("   наведённого уровня — насколько он выше мерила, в пунктах.")
    is_octave(k, j) = (k, j) == (2, 1) || (k, j) == (1, 2)
    is_terce(k, j) = (k, j) == (3, 1) || (k, j) == (1, 3)
    # Доля веса носителя, приходящаяся на уровень (k, j).
    доля(l, k, j) = abs(l[k, j]) / sum(abs.(l))
    function transfer(carriers)
        oct, ter, all = 0, 0, 0
        for carrier in carriers, s in SEEDS
            m2 = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = s)
            detune!(m2, 0.08, 900 + s)
            advance_trace!(m2, 60.0, 4.0, flat, copy(carrier), 0.0, true)
            _, _, counts = found_inside(m2, K, 0.02)
            for ((k, j), n) in counts
                all += n
                is_octave(k, j) && (oct += n)
                is_terce(k, j) && (ter += n)
            end
        end
        return oct, ter, all
    end
    function carrier_of(kind, target, seed)
        m = build(modes = 3, link = 0.0, memory = 0.0, band = 0.05, seed = seed)
        detune!(m, 0.08, 901)
        l = ones(K, K)
        if kind === :phase
            advance_trace!(m, 240.0, 16.0, level_bias(K, target, 16.0), l, 0.5, false)
        else
            advance_record!(m, 240.0, 16.0, level_bias(K, target, 16.0), l, 0.5, kind)
        end
        return to_carrier(l)
    end
    function строка(name, carriers, target, мерило)
        oct, ter, all = transfer(carriers)
        д2, д3 = 100 * oct / all, 100 * ter / all
        к2 = target == (1, 2) ? @sprintf(" %+3.0f", д2 - 100 * мерило) : ""
        к3 = target == (1, 3) ? @sprintf(" %+3.0f", д3 - 100 * мерило) : ""
        @printf("   %-14s носитель: 1:2 %.3f  1:3 %.3f  слияние %.3f   → перенос 1:2 %3d из %3d (%2.0f%%%s)   1:3 %3d (%2.0f%%%s)\n",
                name, доля(carriers[1], 2, 1), доля(carriers[1], 3, 1), доля(carriers[1], 1, 1),
                oct, all, д2, к2, ter, д3, к3)
    end
    список = ((:work, "работа"), (:winner, "срабатывание"), (:margin, "отрыв"),
              (:hold, "порог"), (:coherent, "работа в фазе"), (:sharp, "меткость"),
              (:phase, "угол фазы"))
    println()
    окт0, тер0, всего0 = transfer((flat,))
    @printf("   %-14s октава 1:2 %3d из %3d (%2.0f%%)   терция 1:3 %3d из %3d (%2.0f%%)   — мерило без выбора\n",
            "ровный", окт0, всего0, 100 * окт0 / всего0, тер0, всего0, 100 * тер0 / всего0)
    for (проба, target) in (("проба первая: шкала смещена на 1:2.", (1, 2)),
                            ("проба вторая: шкала смещена на 1:3.", (1, 3)))
        println()
        println("   " * проба)
        мерило = target == (1, 2) ? окт0 / всего0 : тер0 / всего0
        for (kind, name) in список
            строка(name, [carrier_of(kind, target, seed) for seed in (1, 2)], target, мерило)
        end
    end
    println("   Что записано — то и переносится. Работа, срабатывание, порог и работа")
    println("   в фазе держат наведённый уровень первым по весу, и он выше мерила в")
    println("   обеих пробах: октава под 1:2 и терция под 1:3; вперёд по доле он")
    println("   выходит всюду, кроме работы в фазе на 1.13 — там октава обгоняет")
    println("   терцию на пункт. Отрыв держит терцию при любой наводке, а октаву не")
    println("   переносит вовсе: под наводкой на терцию перенос терции усиливается")
    println("   лишь потому, что запись и наводка совпали. Меткость держит слияние")
    println("   (1,1) — оно у неё первым по весу, а уровень следом: под наводкой на")
    println("   терцию вперёд всё равно идёт октава, а терция остаётся ниже мерила.")
    println("   Угол фазы держится неустойчиво: носитель почти гасится, и приведение")
    println("   к среднему раздувает остаток в произвол — в обеих пробах вперёд идёт")
    println("   октава, даже когда наведена терция, и сама прибавка к мерилу гуляет:")
    println("   19 пунктов на 1.10 против 1 на 1.13. Цена разборчивости —")
    println("   неспособность выбрать заново: отрыву и меткости достаётся самая")
    println("   большая доля веса в носителе, а выбора в них нет.")

    return nothing
end

main()