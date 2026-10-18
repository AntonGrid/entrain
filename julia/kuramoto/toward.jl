# Entrain. Курамото: обратный перевес — что делает среда, когда тянет обертон К основной.
#
# Глубина прочитана (`depth.jl`, двадцатая запись), и закон там назван точным:
#
#     λ_нуж = c / (s·p·ω₁),   c = D_p − p·D₁,
#
# — тяга перевеса, делённая на написанное и на размер моды. Но назван он там, где
# среда тянет обертон ОТ основной (c > 0): в сверке такие строки отброшены
# (`c <= 0 && return NaN`), и половина строк осталась непроверенной — 315 мест
# сверено, 285 нет. Это и спрашивается здесь: что у моды внутри, когда среда тянет
# её обертон к основной.
#
# Ожидание, названное заранее: раз направление тяги переменилось, переменится и
# знак ответа. Если строка сама складывается к точному отношению, то чинить её
# нечем и не нужно: нужная жёсткость на таких строках должна оказаться
# неположительной — «строй не нужен». И тогда закон «обратна размеру» — не закон
# моды, а закон одной её половины, и это надо назвать числом, а не оговоркой.
#
# Прочитано то же число без отбрасывания: нужная жёсткость у моды одна на оба
# знака перевеса,
#
#     λ_нуж = |c + γ·r₀_p| / (s·p·ω₁) − γ,
#
# и знак перевеса решает, что делает затухание. При c > 0 модуль раскрывается как
# c + γ·r₀_p, и γ уходит целиком (прочитано). При c < 0 он раскрывается как
# γ·r₀_p − |c|: затухание возвращается — и при |c| > γ·r₀_p входит в ответ дважды
# (λ_нуж = |c|/(s·p·ω₁) − 2γ), а при |c| < 2γ·r₀_p нужная жёсткость неположительна:
# перевес не дотягивает до возврата, строй не нужен. Границы
# этого — нуль и минус удвоенный возврат, −2γ·r₀_p ≤ c ≤ 0: строка держится сама,
# только если среда тянет её к основной и не сильнее, чем возврат затухания её же
# написанного. Обе границы ходят вместе с размером: r₀_p = s·p·ω₀₁ растёт с ним,
# а перевес — нет.
#
# У одной моды при этом не одно направление, а оба сразу: у всех тридцати сред
# (шесть зёрен по пяти размерам) нашлись строки и «от», и «к». Мода слышит две
# тяги одновременно, и закон, прочитанный по половине, читает половину моды.
#
# И отдельно — то, чем «строй не нужен» НЕ является: стоячее и проживание здесь
# расходятся. Неположительная нужная жёсткость говорит, где промах СТОИТ при
# λ = 0 (не хуже написанного), а не куда он БЕЖИТ: за минуту при λ = 0 промах
# расходится по обеим половинам одинаково. Значит неположительная нужная
# жёсткость — не разрешение отпустить среду, а признание, что требование на этой
# строке уже выполнено и строю там делать нечего.
#
# Уравнения те же, что в `basic.jl`. Ничего не подгоняется: перевес, написанное и
# сила строки читаются тем же членом, что в шаге.
#
# Запуск:
#
#     julia julia/kuramoto/toward.jl

include(joinpath(@__DIR__, "basic.jl"))

const SEEDS = 1:6
const SIZES = (0.25, 0.5, 1.0, 2.0, 4.0)
const SHARE = 0.01       # написанный процент по имени — 1 %
const BAND = 0.1         # полоса: та же среда, что в `pair.jl`, `written.jl`, `depth.jl`
const LINK = 0.6
const GAMMA = 0.8        # затухание: то же, что в `written.jl` и `depth.jl`
const LAM = 8.0          # жёсткость из записи: аккорд 1 : 2 : 3 держится ею

# --- что читается ---------------------------------------------------------

"""
    chorded(; seed, lam, band, pitch, damping)

Аккорд 1 : 2 : 3 — та же среда, что в `written.jl` и `depth.jl`: три моды, связь
0.6, память 1e-3, затухание 0.8. Написанное — один процент и тот же у всех мод и
вибраций, так что сравнение идёт по одному написанному. `pitch` — размер аккорда:
свои частоты умножены на него, а числа среды — нет.
"""
function chorded(; seed::Int = 1, lam::Float64 = 0.0, band::Float64 = BAND,
                 pitch::Float64 = 1.0, damping::Float64 = GAMMA)
    m = build(modes = 3, band = band, link = LINK, tuning = lam,
              damping = damping, memory = 1e-3, seed = seed)
    m.omega0 .*= pitch
    m.omega .*= pitch
    return write_share!(m, SHARE)
end

"Написанный промах r₀_p = ω₀_p − p·ω₀₁ — то, чем среда написана."
written(m::Medium, i::Int, p::Int) = m.omega0[i, p] - p * m.omega0[i, 1]

"""
    write_share!(medium, s)

Написать аккорд заново с одним процентом: ω₀_p = p·ω₀₁·(1 + s) — у всех мод и
вибраций сразу. Написанное здесь задаётся долей отношения, а не величиной.
"""
function write_share!(m::Medium, s::Float64)
    for i in 1:size(m.omega0, 1), p in 2:size(m.omega0, 2)
        m.omega0[i, p] = p * m.omega[i, 1] * (1 + s)
    end
    m.omega .= m.omega0
    return m
end

"Тяга связи на моду (i, p) — тот же член, что складывается в `step!`."
function drag(m::Medium, i::Int, p::Int)
    d = 0.0
    for j in 1:size(m.omega, 1)
        j == i && continue
        for n in 1:size(m.omega, 2)
            d += m.coupling[i, j] * overlap(m.omega[j, n] - m.omega[i, p], m.band) *
                 sin(m.theta[j, n] - m.theta[i, p])
        end
    end
    return d
end

"Перевес c = D_p − p·D₁ — тот самый, что читает `excess.jl`."
excess(m::Medium, i::Int, p::Int) = drag(m, i, p) - p * drag(m, i, 1)

"Сила строки в долях отношения: |c + γ·r₀_p|/(p·ω₁) — запись равновесия промаха."
function row_force(m::Medium, i::Int, p::Int)
    base = abs(m.omega[i, 1])
    return abs(drag(m, i, p) + m.damping * written(m, i, p) - p * drag(m, i, 1)) / (p * base)
end

"Нужная жёсткость строки: сила, делённая на написанное, без затухания."
row_need(m::Medium, i::Int, p::Int) = row_force(m, i, p) / SHARE - m.damping

"""
    excess_need(m, i, p)

Нужная жёсткость по прежней форме — c/(s·p·ω₁). Читается только там, где среда
тянет обертон от основной: на «подтягивающих» строках эта форма не проверяется
(`NaN`), и в сверке такие места отброшены — именно это здесь и спрашивается.
"""
function excess_need(m::Medium, i::Int, p::Int)
    c = excess(m, i, p)
    c <= 0.0 && return NaN
    return c / (SHARE * p * abs(m.omega[i, 1]))
end

"Уход затухания в нужной жёсткости: два входа γ в одну строку — и один сдвиг."
toward_need(m::Medium, i::Int, p::Int) = row_need(m, i, p)

"Промах строки в долях отношения — то же, что читает `written.jl` и `depth.jl`."
function row_miss(m::Medium, i::Int, p::Int)
    base = abs(m.omega[i, 1])
    base == 0.0 && return 0.0
    return abs(m.omega[i, p] - p * m.omega[i, 1]) / (p * base)
end

"""
    settled(; seed, lam, band, pitch, seconds)

Среда, прожившая время при шаге описания. Сорвавшееся состояние здесь не сбой, а
отметка: читать уже нечего, и оно помечается бесконечностью.
"""
function settled(; seed::Int = 1, lam::Float64 = 0.0, band::Float64 = BAND,
                 pitch::Float64 = 1.0, seconds::Float64 = 60.0, dt::Float64 = DT)
    m = chorded(seed = seed, lam = lam, band = band, pitch = pitch)
    for _ in 1:round(Int, seconds / dt)
        try
            step!(m, dt)
        catch
            fill!(m.omega, Inf)
            fill!(m.theta, Inf)
            return m
        end
        (all(isfinite, m.omega) && all(isfinite, m.theta)) || return m
    end
    return m
end

"Строки (вибрация, обертон) — те места, где у моды есть промах."
rows(m::Medium) = [(i, p) for i in 1:size(m.omega, 1) for p in 2:size(m.omega, 2)]

"""
    carrier(m)

Строка (i, p), где сила наибольшая: её и читает нужная жёсткость моды.
"""
function carrier(m::Medium)
    worst, found = -Inf, (0, 0)
    for (i, p) in rows(m)
        f = row_force(m, i, p)
        f > worst && ((worst, found) = (f, (i, p)))
    end
    return found
end
"""
    main()

Прочитать обратный перевес: что у моды внутри, когда среда тянет её обертон к
основной.
"""
function main()
    println("обратный перевес: то, чего сверка не читала — строки «к»")

    println("\n--- два знака у одной моды " * "-"^23)
    println("   Перевес c = D_p − p·D₁ — тот же, что читает `excess.jl`: строка «от» —")
    println("   среда тянет обертон от основной, «к» — к основной. Написанное у всех одно:")
    println("   1 %, снято с записи, а не с прогона.")
    println("   размер    строк «к»   строй не нужен   |c| > γ·r₀   |c| ≤ γ·r₀")
    both, total = 0, 0
    negs_by_size, free_by_size = Int[], Int[]
    rows_total, toward_total = 0, 0
    for k in SIZES
        neg, free, twice, short = 0, 0, 0, 0
        for s in SEEDS
            m = chorded(seed = s, lam = LAM, band = BAND * k, pitch = k)
            total += 1
            rows_total += length(rows(m))
            has_pos = any(r -> excess(m, r...) > 0.0, rows(m))
            has_neg = any(r -> excess(m, r...) <= 0.0, rows(m))
            has_pos && has_neg && (both += 1)
            for (i, p) in rows(m)
                c = excess(m, i, p)
                c > 0.0 && continue
                neg += 1
                row_need(m, i, p) <= 0.0 && (free += 1)
                abs(c) > GAMMA * written(m, i, p) ? (twice += 1) : (short += 1)
            end
        end
        @printf("   %6.2f   %9d   %14d   %16d   %15d\n", k, neg, free, twice, short)
        push!(negs_by_size, neg)
        push!(free_by_size, free)
        toward_total += neg
    end
    @printf("   У каждой моды — оба направления сразу: сред с обоими знаками %d из %d.\n",
            both, total)
    println("   Мода слышит две тяги одновременно: одни её обертоны среда складывает к")
    println("   точному отношению, другие — разводит. Закон, прочитанный по половине,")
    println("   читает половину моды.")

    println("\n--- нужная жёсткость: одна формула на оба знака " * "-"^15)
    println("   λ_нуж = |c + γ·r₀_p| / (s·p·ω₁) − γ — без отбрасывания строк. При c > 0")
    println("   модуль раскрывается как c + γ·r₀_p и γ уходит целиком (прочитано); при")
    println("   c < 0 — как γ·r₀_p − |c|, и затухание возвращается.")
    println("   размер   γ уходит   перевес > возврата   возврат > перевеса   ветвь к нужной λ   прежней формой")
    worst_branch = 0.0
    for k in SIZES
        gone, twice, short, tot, old_size = 0, 0, 0, 0, 0
        for s in SEEDS
            m = chorded(seed = s, lam = LAM, band = BAND * k, pitch = k)
            for (i, p) in rows(m)
                tot += 1
                base = abs(m.omega[i, 1])
                c, r0 = excess(m, i, p), written(m, i, p)
                need = row_need(m, i, p)
                if c > 0.0
                    gone += 1
                    worst_branch = max(worst_branch, abs(need - c / (SHARE * p * base)))
                elseif abs(c) > GAMMA * r0
                    twice += 1
                    worst_branch = max(worst_branch,
                        abs(need - (abs(c) - 2 * GAMMA * r0) / (SHARE * p * base)))
                else
                    short += 1
                    worst_branch = max(worst_branch,
                        abs(need + abs(c) / (SHARE * p * base)))
                end
                isnan(excess_need(m, i, p)) && (old_size += 1)
            end
        end
        @printf("   %6.2f   %7d   %9d   %16d   %17.2e   %5d из %d\n",
                k, gone, twice, short, worst_branch, old_size, tot)
    end
    println("   Ветви — не оговорки, а сама формула: γ входит в нужную жёсткость дважды —")
    println("   возвратом к написанному в числителе и вычетом в конце. При c > 0 оба входа")
    println("   гасят друг друга, при c < 0 складываются: тем же затуханием строка и")
    println("   чинится, и портится. Прежней формой, c/(s·p·ω₁), прочитана только одна")
    println("   половина строк — остальные сверка отбрасывала.")

    println("\n--- строй не нужен: это про стоячее, а не про проживание " * "-"^8)
    println("   Неположительная нужная жёсткость значит, что требование на строке уже")
    println("   выполнено. Проверяется тем же равновесием промаха при λ = 0:")
    println("   x = |c + γ·r₀_p| / (γ·p·ω₁) в долях отношения.")
    apart, tot, by_border = 0, 0, 0
    for k in SIZES, s in SEEDS
        m = chorded(seed = s, lam = LAM, band = BAND * k, pitch = k)
        for (i, p) in rows(m)
            tot += 1
            by_need = row_need(m, i, p) <= 0.0
            by_state = row_force(m, i, p) / GAMMA <= SHARE
            by_need == by_state || (apart += 1)
            (-2 * GAMMA * written(m, i, p) <= excess(m, i, p) <= 0.0) && (by_border += 1)
        end
    end
    @printf("   «строй не нужен» и «при λ = 0 промах не хуже написанного» — одно и то же: %d из %d\n",
            tot - apart, tot)
    @printf("   и граница у него та же — перевес от нуля до минус удвоенного возврата: %d из %d\n",
            by_border, tot)
    stop_neg, stop_pos = 0.0, 0.0
    for lam in (0.0, LAM)
        wn, wp, over, tot2 = 0.0, 0.0, 0, 0
        for s in SEEDS
            m0 = chorded(seed = s, lam = lam, band = BAND, pitch = 1.0)
            marks = [(excess(m0, i, p) <= 0.0, i, p) for (i, p) in rows(m0)]
            mm = settled(seed = s, lam = lam, band = BAND, pitch = 1.0)
            for (isneg, i, p) in marks
                d = row_miss(mm, i, p)
                isneg ? (wn = max(wn, d)) : (wp = max(wp, d))
                tot2 += 1
                d > SHARE && (over += 1)
            end
        end
        if lam == 0.0
            stop_neg, stop_pos = wn, wp
        end
        @printf("   λ = %-4.1f  через минуту промах: «к» до %6.3f %%, «от» до %6.3f %%; хуже написанного %d из %d\n",
                lam, 100 * wn, 100 * wp, over, tot2)
    end
    @printf("   При λ = 0 промах расходится по обеим половинам одинаково: %.1f %% и %.1f %%\n",
            100 * stop_neg, 100 * stop_pos)
    println("   против написанного 1 %. Модуль не делает строку держащейся — он говорит,")
    println("   где промах СТОИТ, а не куда он БЕЖИТ. Неположительная нужная жёсткость — не")
    println("   разрешение отпустить среду, а признание, что строю на этой строке нечего")
    println("   делать: требование выполнено само.")

    println("\n--- носитель " * "-"^32)
    nocar, negcar = 0, 0
    cmin, cmax = Inf, -Inf
    neg_need = Float64[]
    for k in SIZES, s in SEEDS
        m = chorded(seed = s, lam = LAM, band = BAND * k, pitch = k)
        (i, p) = carrier(m)
        c = excess(m, i, p)
        cmin, cmax = min(cmin, c), max(cmax, c)
        if c <= 0.0
            negcar += 1
            push!(neg_need, row_need(m, i, p))
        else
            nocar += 1
        end
    end
    @printf("   по наибольшей силе строка «от» %d раз, «к» %d раз; перевес носителя %+.4f … %+.4f\n",
            nocar, negcar, cmin, cmax)
    if isempty(neg_need)
        @printf("   носитель ни разу не «к»: у всех %d сред нужная λ моды читается по тяге\n", nocar)
        println("   перевеса, и затухание в неё входит одним входом — тем же, что гасит второй.")
    else
        @printf("   нужная λ на «к»-носителях: %+.2f … %+.2f — «нужная жёсткость моды» читается\n",
                minimum(neg_need), maximum(neg_need))
        @printf("   и по обратному перевесу (%d сред из %d): затухание входит в неё дважды, а размер\n",
                negcar, nocar + negcar)
        println("   в той же мере её не уменьшает. Одно число, два механизма: пока среда разводит,")
        println("   тяга и размер делят его; когда среда складывает, к нему возвращается затухание.")
    end

    println("\n--- закон «обратна размеру» и его область " * "-"^17)
    key_seed, key_row = 0, (0, 0)
    for s in SEEDS
        m1 = chorded(seed = s, lam = LAM, band = BAND, pitch = 1.0)
        for (i, p) in rows(m1)
            c = excess(m1, i, p)
            if c < 0.0 && abs(c) > GAMMA * written(m1, i, p)
                (key_seed, key_row) = (s, (i, p))
                break
            end
        end
        key_seed != 0 && break
    end
    if key_seed == 0
        println("   Ни при одном размере такой строки нет: у всех «к»-строк перевес не больше")
        println("   возврата, и нужная жёсткость на них читается другой ветвью — той самой, где")
        println("   она неположительна и строю делать нечего.")
    else
        (i, p) = key_row
        @printf("   Строка взята у зерна %d при размере 1 и прослежена по размерам: перевес от\n", key_seed)
        println("   размера не зависит, возврат растёт с ним.")
        println("   размер     перевес c      γ·r₀_p     нужная λ    λ·размер   ветвь      разница с формулой")
        for k in SIZES
            m = chorded(seed = key_seed, lam = LAM, band = BAND * k, pitch = k)
            c, r0 = excess(m, i, p), written(m, i, p)
            need = row_need(m, i, p)
            law = abs(c) * k / (SHARE * p * abs(m.omega[i, 1])) - 2 * GAMMA * k
            straight = abs(c) > GAMMA * r0
            gap = straight ? @sprintf("%.2e", abs(need * k - law)) : "—"
            @printf("   %6.2f   %+10.4f   %10.4f   %+10.4f   %+10.4f   %-9s %s\n",
                    k, c, GAMMA * r0, need, need * k,
                    straight ? "прямая" : "обратная", gap)
        end
    end
    println("   Нужная жёсткость и здесь падает с размером — но её произведение на размер не")
    println("   стоит: оно идёт по прямой |c|/(s·p·ω₁) − 2γ·размер. Так там, где перевес больше")
    println("   возврата затухания: тогда γ входит дважды — и возвратом, и вычетом, — а второй")
    println("   вход растёт с размером. Где возврат больше перевеса, второго входа нет, и")
    println("   λ·размер снова стоит (это видно в таблице на обратной ветви); тогда нужная")
    println("   жёсткость неположительна — строю там делать нечего. У «от»-строк размер входит")
    println("   один раз, поэтому там λ·размер стоит всегда (прочитано в `depth.jl`). Граница")
    println("   Граница ветвей — γ·r₀_p — сама ходит с размером: когда γ·r₀_p перерастает перевес,")
    println("   строка уходит в обратную ветвь, а при чуть большем размере её перевес попадает")
    @printf("   в «строй не нужен» (|c| ≤ 2γ·r₀_p) — потому на размере %.2f таких строк %d из %d, а на %.2f — %d.\n",
            SIZES[end], free_by_size[end], negs_by_size[end], SIZES[1], free_by_size[1])

    @printf("\nИтог. Обратного перевеса не бывает без прямого: у каждой моды внутри оба знака\n")
    @printf("сразу, и это не мелочь, а половина строк — %d из %d. Нужная жёсткость у моды\n",
            toward_total, rows_total)
    println("одна на оба знака, λ_нуж = |c + γ·r₀_p|/(s·p·ω₁) − γ: для «от» γ в ней гасится,")
    println("для «к» — складывается, и при перевесе больше возврата она стоит на 2γ ниже")
    println("прежней формы. Там, где возврат затухания перевес пересиливает, строй не нужен")
    println("вовсе: требование уже выполнено, и это читается равновесием, а не прогоном.")
    println("Закон «обратна размеру» назван точным в своей области: он держится там, где в")
    println("нужную жёсткость входит один γ, — и уходит там, где перевес больше возврата")
    println("затухания: там λ·размер падает ровно на 2γ за единицу размера. Глубина этим")
    println("прочитана целиком: у моды внутри место и размер, а у тяги — знак.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

