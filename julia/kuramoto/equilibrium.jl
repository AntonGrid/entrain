# Entrain. Курамото: мера равновесия — сходится ли окно.
#
# В реестре открытого (глава 22) первое место названо так: у равновесия нет
# меры — `g(λ)` снимается наибольшим по окну `T_win`, и числа при разных окнах
# не сравнимы, пока не сказано, что происходит при `T_win → ∞`. Здесь это
# проверяется, а не постулируется.
#
# Что именно снимается. При заданной жёсткости λ среда живёт `WIN_MAX` секунд, и
# на каждом срезе считается предсказание по строкам
#
#     g(t) = max_{i,p>1} [ F_ip(t)/(требование · p·ω_i1) − γ ],
#
# а мера по окну — максимум по времени:
#
#     g_T = max_{0 ≤ t ≤ T} g(t).
#
# По построению `g_T` не убывает с окном: это максимум. «Предел есть» значит не
# «перестало меняться», а «прирост на удвоении окна стал меньше точности, с
# которой мера вообще различима». Точность названа заранее — 1 %, тот же порог,
# каким в книге меряется промах.
#
# Пороги объявлены до прогона, без порогов после.
#
#   1. Насыщение. У всех шести зёрен при обоих λ (16 и 48) прирост
#      g_480/g_240 − 1 и g_240/g_120 − 1 меньше 1e-2. Тогда мера по окну
#      установилась на 240 секундах с точностью 1 %, и «наибольшее по окну» —
#      число, а не «тем больше, чем дольше смотрим».
#   2. Рост. Если хотя бы на одном удвоении прирост ≥ 1e-2 и знак его не
#      меняется (g растёт на обоих удвоениях), предел на этих окнах не
#      установлен: мера по окну — не число, и в главе 22 первое место остаётся
#      открытым, но с другим содержанием: не «нужно назвать окно», а «окна не
#      хватает ни при каком T».
#   3. Промах. Тот же вопрос для самого промаха: x_480 против x_240 и x_240
#      против x_120. Промах не максимум, он может и падать; здесь важно, стоит
#      ли он или идёт. Это отдельный свидетель: если промах стоит, а g растёт,
#      причина в экстремумах силы, а не в том, что аккорд уезжает.
#   4. Итерация по окну. λ* по окну (итерация λ ← g_T(λ) от 16) при окнах 60,
#      120, 240: расхождение λ*_240 и λ*_120 меньше 1 %, у пяти зёрен из шести.
#      Тогда и корень не зависит от окна, а не только g при чужом λ.
#
# Первое чтение (окна до 480 с) порог 1 не выполнило: установились 4 чтения из
# 12, остальные росли, причём промах к 480 с был мал (0.0007…0.027) — аккорд
# стоит, а наибольшее по окну растёт. Отсюда второй прогон, окна до 3840 с, и
# его исходы названы до него:
#
#   5. Насыщение на длинном окне. Прирост g_3840/g_1920 − 1 меньше 1e-2 —
#      мера установилась, просто не на минуте, а на часе.
#   6. Логарифмический рост. Прирост на последнем удвоении меньше, чем на
#      предыдущем, хотя и больше 1e-2 (падает хотя бы на 20 %). Тогда предел
#      есть только в пределе: на любом конечном окне число зависит от окна, и
#      «наибольшее по окну» — не мера, а семейство мер; называть надо окно.
#   7. Расходимость. Прирост не убывает на удвоениях — предела нет.
#
# Второй прогон дал 11 «стоит» из 12 и одно чтение (зерно 4, λ = 48) с ростом
# 84.9 → 267.1. Это не логарифм и не насыщение: максимум по времени — рекорд,
# он приходится на самый большой выброс силы, и всё зависит от того, попал ли
# выброс в окно. Значит «наибольшее по окну» мерой не является по устройству, а
# не по нехватке времени. Отсюда третье чтение — мера без рекорда, и его
# пороги названы до него:
#
#   8. Устойчивая мера. Квантиль 0.999 и среднее по окну (те же срезы, шаг
#      0.1 с): прирост q_3840/q_960 − 1 и mean_3840/mean_960 − 1 меньше 1e-2 у
#      всех двенадцати чтений. Тогда мера равновесия есть, но это не максимум,
#      а квантиль и среднее; в книге «наибольшее по окну» надо заменить.
#      Не так — у равновесия нет меры и в этом смысле, и первое место реестра
#      остаётся открытым целиком.
#
# Итог (печать сводит 6 и 7 в одно слово «растёт», различение — по числам).
# Порог 8 не выполнен: квантиль меняется от −20 % до +173 %, среднее падает на
# 17…76 % (минута — прогрев). Порог 5 — у 10 чтений из 12; у двух оставшихся
# прирост на последнем удвоении больше, чем на предыдущем (0.25 → 1.48 при λ =
# 16 и 0.57 → 2.15 при λ = 48), то есть это 7, расходимость рекорда. Порог 4 не
# выполнен: λ*_240 против λ*_120 сходится у трёх зёрен из шести (42.65 → 67.18,
# 54.59 → 56.99, 44.48 → 47.09 у трёх остальных), так что от окна зависит и
# корень. Мера равновесия не найдена; в книге это записано числом, а не словом.
#
# Уравнения те же, что в `basic.jl`; среда и снятие — те же, что в `self.jl`
# (`tensioned`, `held_force`, `x_over`), поэтому числа сравнимы с уже
# напечатанными там. Ничего не подгоняется: λ задаётся, окно задаётся, число
# снимается.
#
# Запуск:
#
#     julia julia/kuramoto/equilibrium.jl
#     julia julia/kuramoto/equilibrium.jl trace     # только траектории
#     julia julia/kuramoto/equilibrium.jl iterate   # только итерация по окну

include(joinpath(@__DIR__, "self.jl"))

const WIN_MAX = 3840.0                    # докуда живёт среда: час модельного времени
const WIN_SAMPLE = 0.1                    # как часто снимается срез
const WINS = (15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0)
const WIN_NEED = 0.01                     # требование: промах не хуже 1 %
const WIN_LAMS = (16.0, 48.0)             # два чужих λ: те же, что в `self.jl`
const PLATEAU = 1e-2                      # 1 % — порог насыщения, назван до прогона
const WIN_ITER = (60.0, 120.0, 240.0)     # окна итерации
const WIN_STEPS = 30
const WIN_TOL = 1e-6


# --- что читается ---------------------------------------------------------

"Предсказание по строкам в это мгновение: наибольшее по (i, p > 1)."
function g_now(m::Medium, need::Float64)
    best = -Inf
    for i in 1:size(m.omega, 1), p in 2:size(m.omega, 2)
        base = abs(m.omega[i, 1])
        base == 0.0 && continue
        v = held_force(m, i, p) / (need * p * base) - m.damping
        v > best && (best = v)
    end
    return best
end

"""
    trace_g(need, s, lam; t_max)

Среда при заданной жёсткости живёт `t_max` секунд; на каждом срезе снимаются
`g(t)` и промах `x(t)`. Возвращается путь, а не одно число: мера по окну —
свойство пути, и считать её из одного числа нельзя.
"""
function trace_g(need::Float64, s::Int, lam::Float64; t_max::Float64 = WIN_MAX,
                 detune::Float64 = DETUNE)
    m = tensioned(seed = s, tuning = lam, detune = detune)
    ts, gs, xs = Float64[], Float64[], Float64[]
    broke = false
    for n in 1:round(Int, t_max / DT)
        step!(m, DT)
        if !(all(isfinite, m.omega) && all(isfinite, m.theta))
            broke = true
            break
        end
        n % round(Int, WIN_SAMPLE / DT) == 0 || continue
        push!(ts, n * DT)
        push!(gs, g_now(m, need))
        push!(xs, x_over(m))
    end
    return (t = ts, g = gs, x = xs, broke = broke)
end


"Срез окна: рекорд и его время, квантиль 0.999 и среднее — мера без рекорда."
function stat_window(tr::NamedTuple, win::Float64)
    kmax = 0
    for k in eachindex(tr.t)
        tr.t[k] <= win + 1e-9 || break
        kmax = k
    end
    kmax == 0 && return (max = -Inf, t = NaN, q999 = -Inf, mean = NaN)
    seg = @view tr.g[1:kmax]
    mx, imax = findmax(seg)
    return (max = mx, t = tr.t[imax], q999 = quantile(seg, 0.999), mean = mean(seg))
end


# --- чтение первое: насыщается ли окно -------------------------------------

function main_trace()
    println()
    println("Предел окна: g_T при T от $(WINS[1]) до $(WINS[end]) с, требование $(WIN_NEED)")
    println("Порог насыщения назван до прогона: прирост на удвоении окна < $(PLATEAU)")
    println()

    rows = NamedTuple[]
    steady_n = 0
    for lam in WIN_LAMS
        println("   λ = $(lam)")
        @printf("   рекорд   %-6s %-9s %-9s %-9s  %-8s %s\n",
                "зерно", "max_60", "max_960", "max_3840", "t*, с", "исход")
        for s in SEEDS
            tr = trace_g(WIN_NEED, s, lam)
            if tr.broke || isempty(tr.t)
                @printf("   %-6d сорвалось: читать нечего\n", s)
                push!(rows, (s = s, lam = lam, broke = true, verdict = :сорвалось,
                             q999 = (NaN, NaN), mean = (NaN, NaN)))
                continue
            end
            m60 = stat_window(tr, 60.0)
            m960 = stat_window(tr, 960.0)
            m3840 = stat_window(tr, WIN_MAX)
            rmax = m3840.max / m960.max - 1
            verdict = abs(rmax) < PLATEAU ? :стоит : :растёт
            @printf("   %-8s %-6d %-9.4f %-9.4f %-9.4f  %-8.1f %s\n",
                    "", s, m60.max, m960.max, m3840.max, m3840.t,
                    verdict == :стоит ? "стоит" : "растёт")
            push!(rows, (s = s, lam = lam, broke = false, verdict = verdict,
                         q999 = (m60.q999, m3840.q999), mean = (m60.mean, m3840.mean)))
        end
        println()
        @printf("   мера     %-6s %-9s %-9s %-9s %-9s %-9s %-9s %s\n",
                "зерно", "q999_60", "q999_3840", "прирост", "mean_60", "mean_3840", "прирост", "исход")
        for r in rows
            r.lam == lam || continue
            r.broke && continue
            q1, q2 = r.q999
            m1, m2 = r.mean
            dq = q2 / q1 - 1
            dm = m2 / m1 - 1
            steady = abs(dq) < PLATEAU && abs(dm) < PLATEAU
            @printf("   %-8s %-6d %-9.4f %-9.4f %-9.2e %-9.4f %-9.4f %-9.2e %s\n",
                    "", r.s, q1, q2, dq, m1, m2, dm,
                    steady ? "мерой держится" : "не держится")
            steady_n += steady ? 1 : 0
        end
        println()
    end

    live = [r for r in rows if !r.broke]
    stood = count(r -> r.verdict == :стоит, live)
    steady = steady_n
    @printf("   Рекорд: из %d чтений стоят %d, растут %d\n", length(live), stood, length(live) - stood)
    @printf("   Мера: из %d чтений держатся квантилью и средним %d\n", length(live), steady)
    if steady == length(live) && !isempty(live)
        println("      мера равновесия есть, но это не максимум: квантиль и среднее стоят.")
    elseif steady == 0
        println("      меры равновесия нет и без рекорда: ни квантиль, ни среднее не стоят.")
    else
        println("      мера держится не везде: где именно — видно по строкам.")
    end
    return rows
end


# --- чтение второе: зависит ли корень от окна -------------------------------

"""
    g_window(need, s, lam, win)

Одно чтение при окне `win`: среда живёт ровно окно, мера — наибольшее по нему.
"""
function g_window(need::Float64, s::Int, lam::Float64, win::Float64;
                  detune::Float64 = DETUNE)
    m = tensioned(seed = s, tuning = lam, detune = detune)
    best = -Inf
    for n in 1:round(Int, win / DT)
        step!(m, DT)
        if !(all(isfinite, m.omega) && all(isfinite, m.theta))
            return Inf
        end
        n % round(Int, WIN_SAMPLE / DT) == 0 || continue
        v = g_now(m, need)
        v > best && (best = v)
    end
    return best
end

"Итерация λ ← g_T(λ) при заданном окне: путь и предел."
function iterate_window(need::Float64, s::Int, win::Float64; start::Float64 = START)
    lam = start
    path = [lam]
    for _ in 1:WIN_STEPS
        g = g_window(need, s, lam, win)
        (!isfinite(g) || g > CEIL) && return (path = path, lam = Inf, how = :потолок)
        push!(path, g)
        abs(g - lam) < WIN_TOL && return (path = path, lam = g, how = :сошлось)
        lam = g
    end
    return (path = path, lam = lam, how = :не_сошлось)
end


function main_iterate()
    println()
    println("Итерация по окну: λ* при T = $(join(WIN_ITER, ", ")) с, требование $(WIN_NEED)")
    println("Порог назван до прогона: λ*_240 против λ*_120 — расхождение < 1 % у пяти зёрен из шести")
    println()
    @printf("   %-6s %-10s %-10s %-10s  %-12s %s\n",
            "зерно", "λ*_60", "λ*_120", "λ*_240", "расхождение", "исход")
    roots = Dict{Float64,Vector{Float64}}(w => Float64[] for w in WIN_ITER)
    for s in SEEDS
        vals = Dict{Float64,Float64}()
        for w in WIN_ITER
            r = iterate_window(WIN_NEED, s, w)
            vals[w] = r.lam
            push!(roots[w], r.lam)
        end
        a, b = vals[120.0], vals[240.0]
        rel = (isfinite(a) && isfinite(b) && a != 0) ? abs(b - a) / abs(a) : Inf
        @printf("   %-6d %-10.3f %-10.3f %-10.3f  %-12s %s\n",
                s, vals[60.0], a, b,
                isfinite(rel) ? @sprintf("%.2e", rel) : "—",
                isfinite(a) && isfinite(b) ? (rel < PLATEAU ? "стоит" : "идёт") : "сорвалось")
    end
    println()
    for w in (120.0, 240.0)
        v = filter(isfinite, roots[w])
        if !isempty(v)
            @printf("   окно %3.0f с: λ* от %.3f до %.3f, размах %.3f\n",
                    w, minimum(v), maximum(v), maximum(v) - minimum(v))
        end
    end
    return nothing
end


function main(which::Symbol)
    which in (:all, :trace) && main_trace()
    which in (:all, :iterate) && main_iterate()
    return nothing
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? :all : Symbol(ARGS[1]))
end
