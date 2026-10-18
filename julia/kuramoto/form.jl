# Entrain. Слово против записи: глава 23 «Форма» против `step!`.
#
# Это не опыт и не проверка мира. Проверяется согласие формы с записью:
# всякая величина шага названа словом, и всякое слово имеет место в шаге.
# Равенства главы 23 считаются по коду и сравниваются с тем, что даёт шаг.
#
# Равенства эти — про равновесие, а равновесие здесь установившееся движение:
# тяга D гуляет по фазам, и мгновенно равенства не держатся. Поэтому шаг
# делается с замороженными фазами (`frozen_advance!`) — тогда D постоянна,
# и равенства точны. Что заморозка нужна, а не желательна, видно числом ниже.
#
# Запуск:
#
#     julia julia/kuramoto/form.jl

include(joinpath(@__DIR__, "basic.jl"))

const TOL_FORM = 1e-9      # равенство записи: машинный нуль

"Утверждение формы: величина равна тому, что предписывает запись."
function report_form(name::AbstractString, error::Float64)
    ok = error <= TOL_FORM
    @printf("  %-54s %-11s %.2e\n", name, ok ? "сошлось" : "НЕ СОШЛОСЬ", error)
    return ok
end

"Прожить с замороженными фазами: тогда D не гуляет по фазам."
function frozen_advance!(m, seconds; dt = DT)
    for _ in 1:round(Int, seconds / dt)
        θ = copy(m.theta)
        step!(m, dt)
        m.theta .= θ
    end
    return m
end

"Тяга D_ip по определению главы 23: сила между вибрациями, без строя, затухания и мира."
function drag(m, i, p; band = m.band)
    vibs, modes = size(m.omega)
    d = 0.0
    for j in 1:vibs
        j == i && continue
        for n in 1:modes
            s = overlap(m.omega[j, n] - m.omega[i, p], band)
            d += m.coupling[i, j] * s * sin(m.theta[j, n] - m.theta[i, p])
        end
    end
    return d
end

"Промах обертона в долях при своём λ у мира и λ = 16 у среды."
function held(medium_tuning::Float64, world_tuning::Float64)
    w = build(modes = 2, tuning = medium_tuning)
    inside!(w, coupling = zeros(2, 2), tuning = world_tuning, memory = 0.0)
    advance!(w, 60.0)
    return abs(w.omega[1, 2] - 2 * w.omega[1, 1]) / (2 * abs(w.omega[1, 1]))
end

function form()
    results = Bool[]
    println("Слово против записи\n")

    # Равенства главы 23: при замороженных фазах и Π = ∞ тяга D от частот не зависит.
    m = build(modes = 2, tuning = 16.0, memory = 0.0)
    frozen_advance!(m, 200.0)
    D1 = drag(m, 1, 1)
    D2 = drag(m, 1, 2)
    b = m.omega0[1, 1]
    r0 = m.omega0[1, 2] - 2 * m.omega0[1, 1]
    r = m.omega[1, 2] - 2 * m.omega[1, 1]
    f = abs(D2 + m.damping * r0 - 2 * D1) / (2 * abs(m.omega[1, 1]))
    x = abs(r) / (2 * abs(m.omega[1, 1]))

    println("равенства главы 23 (с замороженными фазами)")
    push!(results, report_form("ω₁ − ω₀₁ = D₁/γ",
        abs((m.omega[1, 1] - m.omega0[1, 1]) - D1 / m.damping)))
    push!(results, report_form("r = (D + γ·r₀ − p·D₁)/(λ+γ)",
        abs(r - (D2 + m.damping * r0 - 2 * D1) / (m.tuning + m.damping))))
    push!(results, report_form("r₀ = p·b·s",
        abs(r0 - 2 * b * (m.omega0[1, 2] / (2 * b) - 1))))
    push!(results, report_form("f = (λ+γ)·x — согласие мер",
        abs(f - (m.tuning + m.damping) * x)))

    # Оговорки записи: вырожденные ветви `step!`.
    m0 = build(modes = 2, memory = 1e-3)
    m0.band = 0.0
    W0 = copy(m0.coupling)
    step!(m0, DT)
    mu = build(modes = 2, inner = 0.3, inner_memory = 0.0)
    U0 = copy(mu.inside[1].coupling)
    step!(mu, DT)

    println("\nоговорки записи")
    push!(results, report_form("Π = 0: связь не меняется",
        maximum(abs.(m0.coupling - W0))))
    push!(results, report_form("εᵤ = 0: U не пишется",
        maximum(abs.(mu.inside[1].coupling - U0))))

    # Порядок шага: фаза берёт уже новую частоту, сил в θ̇ нет.
    mp = build(modes = 2, tuning = 8.0)
    θ_old = copy(mp.theta)
    step!(mp, DT)
    println("\nпорядок шага")
    push!(results, report_form("θ ← θ + dt·ω (новая частота)",
        maximum(abs.(mp.theta - (θ_old + DT .* mp.omega)))))

    # Заморозка нужна, а не желательна: без неё то же равенство не держится.
    mf = build(modes = 2, tuning = 16.0, memory = 0.0)
    advance!(mf, 60.0)
    D1f = drag(mf, 1, 1)
    D2f = drag(mf, 1, 2)
    loose = abs((mf.omega[1, 2] - 2 * mf.omega[1, 1]) -
                (D2f + mf.damping * r0 - 2 * D1f) / (mf.tuning + mf.damping))
    println("\nзаморозка нужна")
    @printf("  %-54s %.2e  (число, не вердикт)\n", "без заморозки: r − формула", loose)

    # λ берётся у мира: тот же шаг, а держит то, что задал мир.
    println("\nλ берётся у мира")
    @printf("  %-54s %.4f\n", "промах: λ мира = 0, в среде 16", held(16.0, 0.0))
    @printf("  %-54s %.4f\n", "промах: λ мира = 16, в среде 16", held(16.0, 16.0))

    println()
    if all(results)
        @printf("слово держится записью: %d из %d\n", count(results), length(results))
    else
        @printf("НЕ СОШЛОСЬ: %d из %d\n", length(results) - count(results), length(results))
    end
    return results
end

form()
