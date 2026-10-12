# Entrain. Курамото: то же описание в движении.
#
# Третья форма той же математики. Не опыт с миром, не проверка гипотез,
# не измерения природы. Здесь сказанное словом и записанное уравнением
# просто движется: видно не «сходится/не сходится», а как это происходит.
#
# Записи те же:
#
#     состояние — частота; у вибрации их несколько: аккорд
#     связь     — coupling: как одна вибрация тянет другую
#     резонанс  — отклик по совпадению частот, полоса Π
#     строй     — λ: жёсткость аккорда, держит обертоны в их отношении
#     затухание — γ: возвращение к своим частотам
#     память    — ε: деформация связи тем, что совпало (русло)
#
# Отличие от basic.jl и consistency.jl:
#
#     basic.jl        — читает числа в отдельных точках
#     consistency.jl  — сверяет описание с самим собой (утверждения)
#     simulations.jl  — показывает, как оно движется (ничего не сверяет)
#
# Запуск:
#
#     julia julia/kuramoto/simulations.jl

using Printf

include(joinpath(@__DIR__, "basic.jl"))

# --- что видно, когда среда живёт ----------------------------------------

"Прочитать состояние среды в этот момент."
function timerow(m::Medium, label::AbstractString)
    @printf("   %-9s согласованность %.3f   разброс частот %.3f   связь %.4f\n",
            label, coherence(m), resonance(m), mean(m.coupling))
    return nothing
end

"""
    live!(medium, marks; label = identity, modes = false)

Прожить средой от метки к метке и показать, чем она была в каждой.
Не «до и после», а весь путь: движение видно только в последовательности.
"""
function live!(m::Medium, marks::Vector{Float64}; label = string)
    prev = 0.0
    timerow(m, label(0.0))
    for t in marks
        advance!(m, t - prev)
        prev = t
        timerow(m, label(t))
    end
    return m
end

"""
    scene_wakes()

Среда живёт: как она приходит к общему звучанию.
"""
function scene_wakes()
    println("--- среда оживает " * "-"^52)
    live!(build(), [1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 45.0, 60.0];
          label = t -> @sprintf("%.0f с", t))
    println("   среда не «переходит» в общее звучание: она им становится.")
    return nothing
end

"""
    scene_by_agreement()

Среда с аккордами собирается не в целое, а по совпадениям частот.
Видно, как каждая мода договаривается сама по себе.
"""
function scene_by_agreement()
    println("\n--- сборка по совпадениям (три моды, узкая полоса) " * "-"^14)
    m = build(modes = 3, band = 0.1, tuning = 8.0)
    prev = 0.0
    for t in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0)
        advance!(m, t - prev)
        prev = t
        parts = join([@sprintf("%.3f", partial_coherence(m, p)) for p in 1:3], "  ")
        @printf("   %-6s целое %.3f   по модам: %s\n",
                @sprintf("%.0f с", t), coherence(m), parts)
    end
    println("   целое отстаёт: каждая мода договаривается на своей частоте.")
    return nothing
end

"""
    scene_tuning()

Строй в движении: как обертоны держатся у основной и как уезжают без строя.
Одно и то же событие в двух средах, отличающихся только λ.
"""
function scene_tuning()
    println("\n--- строй: как аккорд держится во времени " * "-"^24)
    held = build(modes = 3, band = 0.1, tuning = 8.0)
    loose = build(modes = 3, band = 0.1, tuning = 0.0)
    println("   время     расхождение обертонов (держится / уезжает)")
    prev = 0.0
    for t in (0.0, 1.0, 5.0, 15.0, 30.0, 60.0)
        advance!(held, t - prev)
        advance!(loose, t - prev)
        prev = t
        @printf("   %-9s %.4f   %.4f\n", @sprintf("%.0f с", t),
                tuning_error(held, 1), tuning_error(loose, 1))
    end
    println("   без строя аккорд не расходится сразу: он расползается.")
    return nothing
end

"""
    scene_groove()

Русло наводится числом проходов. Одно и то же событие повторяется,
и видно, как среда от одного прохода к другому становится другой.
"""
function scene_groove()
    println("\n--- русло наводится числом проходов " * "-"^29)
    m = build()
    done = 0
    @printf("   проходов %-4d связь[1,2] %.6f   согласованность %.3f\n",
            done, m.coupling[1, 2], coherence(m))
    for target in (1, 2, 5, 10, 20, 50, 100, 200)
        while done < target
            perturb!(m, 1, 1.5)
            advance!(m, 6.0)
            done += 1
        end
        @printf("   проходов %-4d связь[1,2] %.6f   согласованность %.3f\n",
                done, m.coupling[1, 2], coherence(m))
    end
    println("   русло не хранит событие: оно становится тем, как среда отзывается.")
    return nothing
end

"""
    scene_event()

Событие расходится по среде: толчок в первой вибрации и отклик в каждой
из десяти через равные промежутки. Видно не «есть след», а след в пути.
"""
function scene_event()
    println("\n--- событие расходится по среде " * "-"^32)
    m = build()
    perturb!(m, 1, 1.5)
    println("   время     отклик в вибрациях 1 … 10")
    prev = 0.0
    for t in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        advance!(m, t - prev)
        prev = t
        parts = join([@sprintf("%.3f", drift(m, i)) for i in 1:N], " ")
        @printf("   %-9s %s\n", @sprintf("%.2f с", t), parts)
    end
    println("   событие не передаётся: среда откликается там, где совпало.")
    return nothing
end

"""
    scene_returns()

Что делает возвращение к своим частотам: одно событие в двух средах,
у одной γ обычная, у другой возвращения нет вовсе.
"""
function scene_returns()
    println("\n--- возвращение: удар, который гаснет и который не гаснет " * "-"^7)
    alive = build()
    frozen = build(damping = 0.0)
    perturb!(alive, 1, 1.5)
    perturb!(frozen, 1, 1.5)
    println("   время     отклонение (с возвращением / без)")
    prev = 0.0
    for t in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
        advance!(alive, t - prev)
        advance!(frozen, t - prev)
        prev = t
        @printf("   %-9s %.4f   %.4f\n", @sprintf("%.1f с", t),
                drift(alive, 1), drift(frozen, 1))
    end
    println("   без возвращения удар не оседает: частота уезжает и дальше блуждает.")
    return nothing
end

# --- чтение ---------------------------------------------------------------

"""
    motion()

Прочитать описание в движении: сцена за сценой, без вопросов к природе.
"""
function motion()
    @printf("десять вибраций, шаг описания %.2f с — то же описание, в движении\n\n", DT)
    scene_wakes()
    scene_by_agreement()
    scene_tuning()
    scene_groove()
    scene_event()
    scene_returns()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    motion()
end
