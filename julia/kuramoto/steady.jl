# Entrain. Курамото: стоячее как мера равновесия — четвёртый кандидат.
#
# В реестре открытого (глава 22) первое место — «у равновесия нет меры»: `g(λ)`
# снимается наибольшим по окну, и числа при разных окнах не сравнимы, пока не
# сказано, что происходит при `T_win → ∞`. Три кандидата уже отпали:
#
#   максимум по окну     — рекорд: он приходится на самый большой выброс силы,
#                          и всё зависит от того, попал ли выброс в окно;
#   квантиль 0.999       — почти тот же рекорд (0.1 % срезов), меняется на
#                          −20…+173 % при удвоении окна;
#   среднее по всему окну — падает на 17…76 %: в него входит прогрев, и он
#                          определяет число.
#
# Общее у всех трёх — статистика берётся по **всему** окну, вместе с прогревом.
# Не проверено четвёртое: **стоячее** — статистика по хвосту окна, там, где
# прогрев уже позади. Если среда стационаризуется, хвост даёт предел; если
# аккорд медленно уходит, хвост поедет вместе с ним. Это и есть вопрос пункта 1:
# не «какое окно назвать», а «есть ли число, к которому хвост идёт».
#
# Пороги объявлены до прогона, без порогов после.
#
#   1. Хвост. Мера — медиана и среднее `g(t)` на последних `H = 30` секундах
#      окна: `g_хв(T) = med{g(t) : T − H ≤ t ≤ T}` (и то же средним).
#   2. Стоячее. Окна `T ∈ {120, 240, 480}` с. Прирост `|g_хв(240)/g_хв(120) − 1|`
#      и `|g_хв(480)/g_хв(240) − 1|` меньше 1 % — и медианой, и средним.
#   3. Зачёт. Так у пяти зёрен из шести на **обоих** λ (16 и 48). Тогда мера
#      равновесия есть, и она — стоячее; «наибольшее по окну» в книге
#      заменяется на «стоячее (хвост 30 с)».
#   4. Корень. Только при зачёте: итерация `λ ← g_хв(λ)` при окнах 60, 120, 240;
#      расхождение `λ*_240` и `λ*_120` меньше 1 % у пяти зёрен из шести. Тогда и
#      корень не зависит от окна, а не только `g` при чужом λ.
#   5. Не зачёт. Мера не найдена и в смысле предела: тогда пункт 1 реестра
#      закрывается признанием — равновесие меры не имеет, `λ*` остаётся
#      семейством по окну, и это записывается числом, а не словом.
#
# Первое чтение (окна до 480 с) не зачтено: стоят 0 из 12, и падение не
# случайное — хвост уходит вниз в 2…9 раз почти везде (29.99 → 13.59 → 11.74;
# 26.78 → 3.66 → 3.44; 50.89 → 40.68 → 5.41). Это не «нет предела», а «предел
# дальше»: среда релаксирует к стоячему медленнее восьми минут, и наибольшее по
# окну — начало перехода, а не выброс. Отсюда второе чтение, окна до часа, и его
# пороги названы до него:
#
#   6. Хвост на часе. Окна 960, 1920, 3840 с, хвост 30 с. Прирост считается
#      относительно, но не строже абсолютного пола 0.05: у стоячего `g` может
#      подойти к нулю, и деление на почти нуль объявило бы «стоит» там, где
#      менять уже нечего.
#   7. Стоячее на часе. `|g_хв(1920) − g_хв(960)|` и `|g_хв(3840) − g_хв(1920)|`
#      меньше `max(1 % · |g_хв|, 0.05)` у пяти зёрен из шести на обоих λ,
#      медианой и средним. Тогда предел есть, но он за минутой: окно равновесия
#      — час, и оно называется, а не подразумевается.
#   8. Не так — мера не найдена и на часе: релаксация длиннее часа, и `λ*`
#      остаётся семейством по окну; пункт 1 закрывается признанием, с числом.
#
# Второе чтение (окна до часа) тоже не зачтено — 0 из 12, — но падения в нём уже
# нет: медианы хвоста стоят около десятка (9.71…12.16 у десяти чтений из
# двенадцати, 14.73 и 10.02 у двух), а между окнами гуляют на ±20…40 %. Первым
# чтением это было прочитано как шум 30-секундного хвоста; третье чтение это
# опровергло. Отсюда третье чтение — не про длину хвоста, а про стационарность,
# и его пороги названы до него:
#
#   9. Стационарность. Два непересекающихся окна по 480 с внутри часа:
#      A = [960, 1440], B = [3360, 3840]. Если |g_B/g_A − 1| < 1 % и медианой, и
#      средним у пяти зёрен из шести на обоих λ — среда после 960 с стационарна,
#      и мера равновесия есть: это среднее по стоячему, а не наибольшее по окну.
#      Тогда пункт 1 закрывается мерой, и её цена называется: час прогрева и
#      480 с усреднения.
#   10. Не так — среда не стационарна и на часе: мера не найдена, `λ*` остаётся
#      семейством по окну, и пункт 1 закрывается признанием, с числом.
#
# Третье чтение не зачтено: 0 из 12, и это не шум, а дрейф. Окна расходятся от
# −69 % до +58 % (23.54 → 7.40 у зерна 4 при λ = 48; 6.67 → 10.52 у зерна 2 при
# λ = 48; 6.97 → 10.12 у зерна 5 при λ = 16), и у девяти чтений из двенадцати
# позднее окно выше раннего — среда подходит к стоячему снизу и к концу часа не
# подходит. Будь это флуктуации, два окна по 480 с сошлись бы: длинное окно шум
# усредняет, а дрейф — нет. Итог: меры равновесия нет ни в одном из четырёх
# смыслов (рекорд, квантиль 0.999, среднее по всему окну, стоячее), `λ*` —
# функция времени наблюдения, и пункт 1 реестра закрыт признанием, с числом
# (тридцать вторая запись).
#
# Уравнения те же, что в `basic.jl`; среда и снятие — те же, что в `equilibrium.jl`
# (`trace_g`, `g_now`), поэтому числа сравнимы с уже напечатанными там. Ничего не
# подгоняется: λ задаётся, окно задаётся, число снимается.
#
# Запуск:
#
#     julia julia/kuramoto/steady.jl          # первое чтение: окна до 480 с
#     julia julia/kuramoto/steady.jl long     # второе чтение: окна до часа
#     julia julia/kuramoto/steady.jl stat     # третье чтение: стационарность на часе

include(joinpath(@__DIR__, "equilibrium.jl"))   # trace_g, g_now, WIN_NEED, SEEDS

const ST_WINS = (120.0, 240.0, 480.0)    # окна, на которых видно, стоит ли хвост
const ST_TAIL = 30.0                     # длина хвоста: прогрев позади
const ST_PLATEAU = 1e-2                  # 1 % — тот же порог, что в книге
const ST_LAMS = (16.0, 48.0)             # два чужих λ: те же, что в `equilibrium.jl`
const ST_NEED_STEADY = 5                 # из шести зёрен
const ST_ITER_WINS = (60.0, 120.0, 240.0)
const ST_ITER_STEPS = 20
const ST_ITER_TOL = 1e-6
const ST_LONG_WINS = (960.0, 1920.0, 3840.0)   # второе чтение: час
const ST_FLOOR = 0.05                          # абсолютный пол: у стоячего g идёт к нулю
const ST_STAT_A = (960.0, 1440.0)              # третье чтение: два окна по 480 с
const ST_STAT_B = (3360.0, 3840.0)

"Хвост окна: медиана и среднее `g` на `[win − h, win]` — стоячее, а не рекорд."
function tail_stat(tr::NamedTuple, win::Float64, h::Float64 = ST_TAIL)
    idx = [k for k in eachindex(tr.t) if tr.t[k] <= win + 1e-9 && tr.t[k] >= win - h - 1e-9]
    isempty(idx) && return (med = NaN, mean = NaN, n = 0)
    seg = @view tr.g[idx]
    return (med = median(seg), mean = mean(seg), n = length(idx))
end


"Стоячее при заданной жёсткости: среда живёт `win` секунд, читается хвост."
function g_steady(need::Float64, s::Int, lam::Float64, win::Float64;
                  detune::Float64 = DETUNE, h::Float64 = ST_TAIL)
    tr = trace_g(need, s, lam; t_max = win, detune = detune)
    (tr.broke || isempty(tr.t)) && return (med = Inf, mean = Inf)
    t = tail_stat(tr, win, h)
    return (med = t.med, mean = t.mean)
end


"Итерация `λ ← g_хв(λ)` при данном окне: путь и предел."
function iterate_steady(need::Float64, s::Int, win::Float64; start::Float64 = START)
    lam = start
    path = [lam]
    for _ in 1:ST_ITER_STEPS
        g = g_steady(need, s, lam, win).med
        (!isfinite(g) || g > CEIL) && return (path = path, lam = Inf, how = :потолок)
        push!(path, g)
        abs(g - lam) < ST_ITER_TOL && return (path = path, lam = g, how = :сошлось)
        lam = g
    end
    return (path = path, lam = lam, how = :не_сошлось)
end



function main_steady()
    println()
    println("Стоячее как мера: хвост $(ST_TAIL) с, окна $(join(ST_WINS, ", ")) с")
    println("Пороги названы до прогона: прирост на удвоении окна < $(ST_PLATEAU); ",
            "зачёт — у $(ST_NEED_STEADY) зёрен из $(length(SEEDS)) на обоих λ, медианой и средним")
    println()

    rows = NamedTuple[]
    for lam in ST_LAMS
        println("   λ = $(lam)")
        @printf("   %-6s %-9s %-9s %-9s %-9s %-9s %-9s  %-8s %s\n",
                "зерно", "med_120", "med_240", "med_480", "mean_120", "mean_240", "mean_480",
                "приросты", "исход")
        for s in SEEDS
            tr = trace_g(WIN_NEED, s, lam; t_max = ST_WINS[end])
            if tr.broke || isempty(tr.t)
                @printf("   %-6d сорвалось: читать нечего\n", s)
                push!(rows, (s = s, lam = lam, broke = true, med = false, mean = false))
                continue
            end
            a = tail_stat(tr, ST_WINS[1])
            b = tail_stat(tr, ST_WINS[2])
            c = tail_stat(tr, ST_WINS[3])
            dm1, dm2 = b.med / a.med - 1, c.med / b.med - 1
            dn1, dn2 = b.mean / a.mean - 1, c.mean / b.mean - 1
            med_ok = abs(dm1) < ST_PLATEAU && abs(dm2) < ST_PLATEAU
            mean_ok = abs(dn1) < ST_PLATEAU && abs(dn2) < ST_PLATEAU
            @printf("   %-6d %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f  %-8s %s\n",
                    s, a.med, b.med, c.med, a.mean, b.mean, c.mean,
                    @sprintf("%+.1e/%+.1e", dm2, dn2),
                    med_ok && mean_ok ? "стоит" : (med_ok ? "медиана" : (mean_ok ? "среднее" : "идёт")))
            push!(rows, (s = s, lam = lam, broke = false, med = med_ok, mean = mean_ok))
        end
        println()
    end

    live = [r for r in rows if !r.broke]
    both = count(r -> r.med && r.mean, live)
    @printf("   Стоячее: из %d чтений стоят медианой и средним %d\n", length(live), both)
    ok = both >= 2 * ST_NEED_STEADY
    if ok
        println("      мера равновесия есть, и это стоячее: хвост 30 с — число, а не семейство.")
    elseif both == 0
        println("      меры равновесия нет и в смысле предела: хвост идёт вместе с аккордом.")
    else
        println("      стоячее держится не везде: где именно — видно по строкам.")
    end
    return (rows = rows, ok = ok)
end


function main_iterate_steady()
    println()
    println("Итерация по стоячему: λ* при окнах $(join(ST_ITER_WINS, ", ")) с, требование $(WIN_NEED)")
    println("Порог назван до прогона: λ*_240 против λ*_120 — расхождение < $(ST_PLATEAU) у $(ST_NEED_STEADY) зёрен из $(length(SEEDS))")
    println()
    @printf("   %-6s %-10s %-10s %-10s  %-12s %s\n",
            "зерно", "λ*_60", "λ*_120", "λ*_240", "расхождение", "исход")
    roots = Dict{Float64,Vector{Float64}}(w => Float64[] for w in ST_ITER_WINS)
    stood = 0
    for s in SEEDS
        vals = Dict{Float64,Float64}()
        for w in ST_ITER_WINS
            r = iterate_steady(WIN_NEED, s, w)
            vals[w] = r.lam
            push!(roots[w], r.lam)
        end
        a, b = vals[120.0], vals[240.0]
        rel = (isfinite(a) && isfinite(b) && a != 0) ? abs(b - a) / abs(a) : Inf
        stood += (isfinite(rel) && rel < ST_PLATEAU) ? 1 : 0
        @printf("   %-6d %-10.3f %-10.3f %-10.3f  %-12s %s\n",
                s, vals[60.0], a, b,
                isfinite(rel) ? @sprintf("%.2e", rel) : "—",
                isfinite(a) && isfinite(b) ? (rel < ST_PLATEAU ? "стоит" : "идёт") : "сорвалось")
    end
    println()
    for w in (120.0, 240.0)
        v = filter(isfinite, roots[w])
        isempty(v) || @printf("   окно %3.0f с: λ* от %.3f до %.3f, размах %.3f\n",
                              w, minimum(v), maximum(v), maximum(v) - minimum(v))
    end
    @printf("   Корень: λ*_240 против λ*_120 стоит у %d зёрен из %d\n", stood, length(SEEDS))
    return stood
end


"Стоит ли хвост: относительный порог 1 %, но не строже абсолютного пола."
function stands_at(g1::Float64, g2::Float64)
    (isfinite(g1) && isfinite(g2)) || return false
    return abs(g2 - g1) < max(ST_PLATEAU * abs(g1), ST_FLOOR)
end


function main_steady_long()
    println()
    println("Стоячее на часе: хвост $(ST_TAIL) с, окна $(join(ST_LONG_WINS, ", ")) с")
    println("Пороги названы до прогона: |Δg| < max(1 % · |g|, $(ST_FLOOR)); ",
            "зачёт — у $(ST_NEED_STEADY) зёрен из $(length(SEEDS)) на обоих λ, медианой и средним")
    println()

    rows = NamedTuple[]
    for lam in ST_LAMS
        println("   λ = $(lam)")
        @printf("   %-6s %-9s %-9s %-9s %-9s %-9s %-9s  %-8s %s\n",
                "зерно", "med_960", "med_1920", "med_3840", "mean_960", "mean_1920", "mean_3840",
                "приросты", "исход")
        for s in SEEDS
            tr = trace_g(WIN_NEED, s, lam; t_max = ST_LONG_WINS[end])
            if tr.broke || isempty(tr.t)
                @printf("   %-6d сорвалось: читать нечего\n", s)
                push!(rows, (s = s, lam = lam, broke = true, med = false, mean = false))
                continue
            end
            a = tail_stat(tr, ST_LONG_WINS[1])
            b = tail_stat(tr, ST_LONG_WINS[2])
            c = tail_stat(tr, ST_LONG_WINS[3])
            med_ok = stands_at(a.med, b.med) && stands_at(b.med, c.med)
            mean_ok = stands_at(a.mean, b.mean) && stands_at(b.mean, c.mean)
            @printf("   %-6d %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f  %-8s %s\n",
                    s, a.med, b.med, c.med, a.mean, b.mean, c.mean,
                    @sprintf("%+.1e/%+.1e", c.med - b.med, c.mean - b.mean),
                    med_ok && mean_ok ? "стоит" : (med_ok ? "медиана" : (mean_ok ? "среднее" : "идёт")))
            push!(rows, (s = s, lam = lam, broke = false, med = med_ok, mean = mean_ok))
        end
        println()
    end

    live = [r for r in rows if !r.broke]
    both = count(r -> r.med && r.mean, live)
    @printf("   Стоячее на часе: из %d чтений стоят медианой и средним %d\n", length(live), both)
    if both >= 2 * ST_NEED_STEADY
        println("      предел есть, но он за минутой: окно равновесия — час, и оно называется.")
    elseif both == 0
        println("      меры нет и на часе: релаксация длиннее часа, λ* остаётся семейством по окну.")
    else
        println("      на часе стоит не везде: где именно — видно по строкам.")
    end
    return (rows = rows, both = both)
end


"Окно между двумя временами: медиана и среднее `g` на `[t1, t2]`."
function window_stat(tr::NamedTuple, t1::Float64, t2::Float64)
    idx = [k for k in eachindex(tr.t) if tr.t[k] >= t1 - 1e-9 && tr.t[k] <= t2 + 1e-9]
    isempty(idx) && return (med = NaN, mean = NaN, n = 0)
    seg = @view tr.g[idx]
    return (med = median(seg), mean = mean(seg), n = length(idx))
end


function main_steady_stat()
    println()
    println("Стационарность: два окна по $(ST_STAT_B[2] - ST_STAT_B[1]) с внутри часа — ",
            "A = [$(ST_STAT_A[1]), $(ST_STAT_A[2])], B = [$(ST_STAT_B[1]), $(ST_STAT_B[2])]")
    println("Пороги названы до прогона: |g_B/g_A − 1| < $(ST_PLATEAU) и медианой, и средним; ",
            "зачёт — у $(ST_NEED_STEADY) зёрен из $(length(SEEDS)) на обоих λ")
    println()

    rows = NamedTuple[]
    for lam in ST_LAMS
        println("   λ = $(lam)")
        @printf("   %-6s %-9s %-9s %-9s %-9s %-9s %-9s  %s\n",
                "зерно", "med_A", "med_B", "прирост", "mean_A", "mean_B", "прирост", "исход")
        for s in SEEDS
            tr = trace_g(WIN_NEED, s, lam; t_max = ST_STAT_B[2])
            if tr.broke || isempty(tr.t)
                @printf("   %-6d сорвалось: читать нечего\n", s)
                push!(rows, (s = s, lam = lam, broke = true, ok = false))
                continue
            end
            a = window_stat(tr, ST_STAT_A[1], ST_STAT_A[2])
            b = window_stat(tr, ST_STAT_B[1], ST_STAT_B[2])
            dm = b.med / a.med - 1
            dn = b.mean / a.mean - 1
            ok = abs(dm) < ST_PLATEAU && abs(dn) < ST_PLATEAU
            @printf("   %-6d %-9.4f %-9.4f %-9.2e %-9.4f %-9.4f %-9.2e  %s\n",
                    s, a.med, b.med, dm, a.mean, b.mean, dn,
                    ok ? "стационарно" : "идёт")
            push!(rows, (s = s, lam = lam, broke = false, ok = ok))
        end
        println()
    end

    live = [r for r in rows if !r.broke]
    good = count(r -> r.ok, live)
    @printf("   Стационарность: из %d чтений совпали на 480 с %d\n", length(live), good)
    if good >= 2 * ST_NEED_STEADY
        println("      среда после 960 с стационарна: мера равновесия есть — среднее по стоячему,")
        println("      ценой часа прогрева и 480 с усреднения; «наибольшее по окну» заменяется им.")
    elseif good == 0
        println("      среда не стационарна и на часе: мера не найдена, λ* остаётся семейством по окну.")
    else
        println("      стационарность держится не везде: где именно — видно по строкам.")
    end
    return (rows = rows, good = good)
end



if abspath(PROGRAM_FILE) == @__FILE__
    arg = isempty(ARGS) ? "" : ARGS[1]
    if arg == "long"
        main_steady_long()
    elseif arg == "stat"
        main_steady_stat()
    else
        res = main_steady()
        res.ok && main_iterate_steady()
    end
end

