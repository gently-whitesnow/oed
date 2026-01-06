#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №4. Вариант 7. Часть 1
Однофакторный регрессионный анализ

Задача: На основе массива экспериментальных данных построить уравнение регрессии
в виде алгебраического полинома второй степени, проверить его адекватность
и значимость коэффициентов регрессии. Расчёты произвести в скалярной и матричной форме.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Настройка поддержки русского языка в графиках
plt.rcParams['font.family'] = 'DejaVu Sans'

# ========== ШАГ 1: ИСХОДНЫЕ ДАННЫЕ ==========

print("ШАГ 1: ИСХОДНЫЕ ДАННЫЕ")


# Данные для варианта 7
x = np.array([-2, 0, 1, 2, 3], dtype=float)
y = np.array([-2, -4, 1, 10, 21], dtype=float)
n = len(x)
alpha = 0.01  # Уровень значимости

print(f"\nЭкспериментальные данные (n = {n}):")
print(f"x = {x}")
print(f"y = {y}")
print(f"\nУровень значимости α = {alpha}")

# Визуализация исходных данных
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', s=100, zorder=5, label='Экспериментальные данные')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Исходные экспериментальные данные')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('data_plot.png', dpi=150, bbox_inches='tight')

# ========== ШАГ 2: ПОСТРОЕНИЕ СИСТЕМЫ НОРМАЛЬНЫХ УРАВНЕНИЙ (СКАЛЯРНАЯ ФОРМА) ==========

print("ШАГ 2: ПОСТРОЕНИЕ СИСТЕМЫ НОРМАЛЬНЫХ УРАВНЕНИЙ (СКАЛЯРНАЯ ФОРМА)")
print("\nМодель регрессии: y = b₀ + b₁·x + b₂·x²")
print("\nДля построения системы нормальных уравнений необходимо вычислить суммы:")

# Вычисление необходимых сумм
sum_1 = n  # сумма единиц
sum_x = np.sum(x)
sum_x2 = np.sum(x**2)
sum_x3 = np.sum(x**3)
sum_x4 = np.sum(x**4)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2y = np.sum(x**2 * y)

print(f"\n∑1   = {sum_1}")
print(f"∑x   = {sum_x}")
print(f"∑x²  = {sum_x2}")
print(f"∑x³  = {sum_x3}")
print(f"∑x⁴  = {sum_x4}")
print(f"∑y   = {sum_y}")
print(f"∑xy  = {sum_xy}")
print(f"∑x²y = {sum_x2y}")

print("\nСистема нормальных уравнений:")
print(f"{sum_1}·b₀ + {sum_x}·b₁ + {sum_x2}·b₂ = {sum_y}")
print(f"{sum_x}·b₀ + {sum_x2}·b₁ + {sum_x3}·b₂ = {sum_xy}")
print(f"{sum_x2}·b₀ + {sum_x3}·b₁ + {sum_x4}·b₂ = {sum_x2y}")

# Формирование матрицы системы для скалярного решения
A_scalar = np.array([
    [sum_1, sum_x, sum_x2],
    [sum_x, sum_x2, sum_x3],
    [sum_x2, sum_x3, sum_x4]
])
b_scalar = np.array([sum_y, sum_xy, sum_x2y])

print("\nМатричная форма системы A·b = c:")
print("Матрица A:")
print(A_scalar)
print("\nВектор c:")
print(b_scalar)

# Решение системы (скалярная форма)
coefficients_scalar = np.linalg.solve(A_scalar, b_scalar)
b0_scalar, b1_scalar, b2_scalar = coefficients_scalar

print("\nРешение системы нормальных уравнений (скалярная форма):")
print(f"b₀ = {b0_scalar:.6f}")
print(f"b₁ = {b1_scalar:.6f}")
print(f"b₂ = {b2_scalar:.6f}")

print(f"\nУравнение регрессии: y = {b0_scalar:.6f} + {b1_scalar:.6f}·x + {b2_scalar:.6f}·x²")

# ========== ШАГ 3: РЕШЕНИЕ В МАТРИЧНОЙ ФОРМЕ ==========

print("ШАГ 3: РЕШЕНИЕ В МАТРИЧНОЙ ФОРМЕ")

print("\nВ матричной форме уравнение регрессии представляется как:")
print("Y = X·B, где")
print("Y - вектор наблюдений")
print("X - матрица планирования")
print("B - вектор коэффициентов регрессии")

# Матрица планирования
X_matrix = np.column_stack([np.ones(n), x, x**2])
print("\nМатрица планирования X:")
print(X_matrix)

print("\nВектор наблюдений Y:")
print(y)

# Оценка коэффициентов по МНК: B = (X^T·X)^(-1)·X^T·Y
X_T = X_matrix.T
XTX = X_T @ X_matrix
XTY = X_T @ y

print("\nМатрица X^T·X:")
print(XTX)

print("\nВектор X^T·Y:")
print(XTY)

# Обратная матрица
XTX_inv = np.linalg.inv(XTX)
print("\nОбратная матрица (X^T·X)^(-1):")
print(XTX_inv)

# Коэффициенты регрессии
coefficients_matrix = XTX_inv @ XTY
b0_matrix, b1_matrix, b2_matrix = coefficients_matrix

print("\nОценки коэффициентов регрессии (матричная форма):")
print(f"b₀ = {b0_matrix:.6f}")
print(f"b₁ = {b1_matrix:.6f}")
print(f"b₂ = {b2_matrix:.6f}")

print(f"\nУравнение регрессии: y = {b0_matrix:.6f} + {b1_matrix:.6f}·x + {b2_matrix:.6f}·x²")

# Проверка совпадения результатов
print("\nПроверка совпадения результатов скалярной и матричной форм:")
print(f"Разница в коэффициентах: {np.max(np.abs(coefficients_scalar - coefficients_matrix)):.2e}")

# Используем коэффициенты из матричной формы для дальнейших расчетов
b = coefficients_matrix

# ========== ШАГ 4: ВЫЧИСЛЕНИЕ РАСЧЕТНЫХ ЗНАЧЕНИЙ И ОСТАТКОВ ==========

print("ШАГ 4: ВЫЧИСЛЕНИЕ РАСЧЕТНЫХ ЗНАЧЕНИЙ И ОСТАТКОВ")


# Расчетные значения
y_pred = X_matrix @ b

print("\nТаблица данных:")
print(f"{'i':<5} {'x':<8} {'y (эксп.)':<12} {'y (расч.)':<12} {'Остаток':<12}")
print("-" * 60)
for i in range(n):
    residual = y[i] - y_pred[i]
    print(f"{i+1:<5} {x[i]:<8.2f} {y[i]:<12.2f} {y_pred[i]:<12.6f} {residual:<12.6f}")

# Остатки
residuals = y - y_pred

# Сумма квадратов остатков
SS_res = np.sum(residuals**2)

# Сумма квадратов относительно среднего
y_mean = np.mean(y)
SS_tot = np.sum((y - y_mean)**2)

# Объясненная сумма квадратов
SS_reg = np.sum((y_pred - y_mean)**2)

print(f"\nСреднее значение y: ȳ = {y_mean:.6f}")
print(f"\nСумма квадратов остатков: SS_res = {SS_res:.6f}")
print(f"Сумма квадратов относительно среднего: SS_tot = {SS_tot:.6f}")
print(f"Объясненная сумма квадратов: SS_reg = {SS_reg:.6f}")
print(f"Проверка: SS_tot = SS_reg + SS_res: {SS_tot:.6f} = {SS_reg + SS_res:.6f}")

# Коэффициент детерминации
R2 = SS_reg / SS_tot if SS_tot != 0 else 0
print(f"\nКоэффициент детерминации R² = {R2:.6f}")
print(f"Это означает, что {R2*100:.2f}% вариации y объясняется моделью")

# ========== ШАГ 5: ПРОВЕРКА АДЕКВАТНОСТИ ПО КРИТЕРИЮ ФИШЕРА ==========

print("ШАГ 5: ПРОВЕРКА АДЕКВАТНОСТИ ПО КРИТЕРИЮ ФИШЕРА")

# Количество коэффициентов
p = 3  # b0, b1, b2

# Степени свободы
df_reg = p - 1  # Степени свободы регрессии (без свободного члена для F-теста)
df_res = n - p  # Степени свободы остатков

print(f"\nКоличество наблюдений: n = {n}")
print(f"Количество коэффициентов: p = {p}")
print(f"Степени свободы регрессии: df_reg = {df_reg}")
print(f"Степени свободы остатков: df_res = {df_res}")

# Средние квадраты
MS_reg = SS_reg / df_reg if df_reg > 0 else 0
MS_res = SS_res / df_res if df_res > 0 else 0

print(f"\nСредний квадрат регрессии: MS_reg = {MS_reg:.6f}")
print(f"Средний квадрат остатков: MS_res = {MS_res:.6f}")

# F-статистика
F_stat = MS_reg / MS_res if MS_res > 0 else float('inf')

# Критическое значение F-распределения
F_crit = stats.f.ppf(1 - alpha, df_reg, df_res)

print(f"\nF-статистика: F = {F_stat:.6f}")
print(f"Критическое значение F_{{{alpha}, {df_reg}, {df_res}}} = {F_crit:.6f}")

# P-значение
p_value_F = 1 - stats.f.cdf(F_stat, df_reg, df_res)
print(f"P-значение: p = {p_value_F:.6f}")

print("\nВывод:")
if F_stat > F_crit:
    print(f"F ({F_stat:.4f}) > F_crit ({F_crit:.4f})")
    print(f"Уравнение регрессии АДЕКВАТНО экспериментальным данным (α = {alpha})")
else:
    print(f"F ({F_stat:.4f}) <= F_crit ({F_crit:.4f})")
    print(f"Уравнение регрессии НЕ АДЕКВАТНО экспериментальным данным (α = {alpha})")

# ========== ШАГ 6: ПРОВЕРКА ЗНАЧИМОСТИ КОЭФФИЦИЕНТОВ ПО КРИТЕРИЮ СТЬЮДЕНТА ==========

print("ШАГ 6: ПРОВЕРКА ЗНАЧИМОСТИ КОЭФФИЦИЕНТОВ ПО КРИТЕРИЮ СТЬЮДЕНТА")

# Оценка дисперсии остатков
s2 = SS_res / df_res if df_res > 0 else 0
s = np.sqrt(s2)

print(f"\nОценка среднеквадратичного отклонения остатков: s = {s:.6f}")
print(f"Оценка дисперсии остатков: s² = {s2:.6f}")

# Ковариационная матрица коэффициентов
cov_matrix = s2 * XTX_inv
print("\nКовариационная матрица коэффициентов:")
print(cov_matrix)

# Стандартные ошибки коэффициентов
se = np.sqrt(np.diag(cov_matrix))

print("\nСтандартные ошибки коэффициентов:")
for i, label in enumerate(['b₀', 'b₁', 'b₂']):
    print(f"SE({label}) = {se[i]:.6f}")

# t-статистики
t_stats = b / se

# Критическое значение t-распределения
t_crit = stats.t.ppf(1 - alpha/2, df_res)

print(f"\nКритическое значение t_{{{alpha/2}, {df_res}}} = {t_crit:.6f}")

print("\nПроверка значимости коэффициентов:")
print(f"{'Коэфф.':<10} {'Значение':<15} {'Ст. ошибка':<15} {'t-стат.':<15} {'|t| > t_crit':<15} {'Значим':<10}")
print("-" * 85)

significant = []
for i, label in enumerate(['b₀', 'b₁', 'b₂']):
    is_significant = abs(t_stats[i]) > t_crit
    significant.append(is_significant)
    print(f"{label:<10} {b[i]:<15.6f} {se[i]:<15.6f} {t_stats[i]:<15.6f} "
          f"{abs(t_stats[i]) > t_crit!s:<15} {'ДА' if is_significant else 'НЕТ':<10}")

# P-значения для коэффициентов
print("\nP-значения для коэффициентов:")
for i, label in enumerate(['b₀', 'b₁', 'b₂']):
    p_value_t = 2 * (1 - stats.t.cdf(abs(t_stats[i]), df_res))
    print(f"{label}: p = {p_value_t:.6f}")

# Определение незначимых коэффициентов
candidate_indices = list(range(1, p))  # проверяем на исключение только b1..b(p-1)
insignificant_indices = [i for i in candidate_indices if not significant[i]]
insignificant_labels = [['b₀', 'b₁', 'b₂'][i] for i in insignificant_indices]

if insignificant_indices:
    print(f"\nНезначимые коэффициенты среди b₁..b₂: {', '.join(insignificant_labels)}")
else:
    print("\nСреди b₁..b₂ все коэффициенты значимы (b₀ не исключаем по условию).")

# ========== ШАГ 7: ПОВТОРНАЯ ПРОВЕРКА ПОСЛЕ ИСКЛЮЧЕНИЯ НЕЗНАЧИМЫХ КОЭФФИЦИЕНТОВ ==========

print("ШАГ 7: ПОВТОРНАЯ ПРОВЕРКА ПОСЛЕ ИСКЛЮЧЕНИЯ НЕЗНАЧИМЫХ КОЭФФИЦИЕНТОВ")

# По умолчанию считаем, что итоговая модель — полная
final_indices = list(range(p))
b_final = b.copy()
X_final = X_matrix
y_pred_final = y_pred.copy()
residuals_final = residuals.copy()
SS_res_final = SS_res
SS_reg_final = SS_reg
R2_final = R2
df_reg_final = df_reg
df_res_final = df_res
F_stat_final = F_stat
F_crit_final = F_crit

if insignificant_indices:
    print(f"\nИсключаем незначимые коэффициенты (кроме b₀): {', '.join(insignificant_labels)}")

    # Оставляем b0 всегда, плюс те, что значимы среди b1..b2
    significant_indices = [0] + [i for i in range(1, p) if i not in insignificant_indices]
    X_reduced = X_matrix[:, significant_indices]

    print(f"\nНовая матрица планирования (размер {X_reduced.shape}):")
    print(X_reduced)

    # Пересчитываем коэффициенты
    XTX_reduced = X_reduced.T @ X_reduced
    XTY_reduced = X_reduced.T @ y
    XTX_inv_reduced = np.linalg.inv(XTX_reduced)
    b_reduced = XTX_inv_reduced @ XTY_reduced

    print("\nНовые оценки коэффициентов:")
    significant_labels_list = [['b₀', 'b₁', 'b₂'][i] for i in significant_indices]
    for i, label in enumerate(significant_labels_list):
        print(f"{label} = {b_reduced[i]:.6f}")

    # Новые расчётные значения
    y_pred_reduced = X_reduced @ b_reduced

    # Новая модель (печать)
    print("\nУпрощенная модель:")
    model_terms = []
    for i, idx in enumerate(significant_indices):
        if idx == 0:
            model_terms.append(f"{b_reduced[i]:.6f}")
        elif idx == 1:
            model_terms.append(f"{b_reduced[i]:.6f}·x")
        elif idx == 2:
            model_terms.append(f"{b_reduced[i]:.6f}·x²")
    print(f"y = {' + '.join(model_terms)}")

    # Новые остатки и суммы квадратов (ВАЖНО: b0 есть → можно использовать те же формулы SS_reg = Σ(ŷ-ȳ)^2)
    residuals_reduced = y - y_pred_reduced
    SS_res_reduced = np.sum(residuals_reduced**2)
    SS_reg_reduced = np.sum((y_pred_reduced - y_mean)**2)

    R2_reduced = SS_reg_reduced / SS_tot if SS_tot != 0 else 0

    # Новая проверка адекватности (F-тест)
    p_reduced = len(significant_indices)
    df_reg_reduced = p_reduced - 1
    df_res_reduced = n - p_reduced

    MS_reg_reduced = SS_reg_reduced / df_reg_reduced if df_reg_reduced > 0 else 0
    MS_res_reduced = SS_res_reduced / df_res_reduced if df_res_reduced > 0 else 0

    F_stat_reduced = MS_reg_reduced / MS_res_reduced if MS_res_reduced > 0 else float('inf')
    F_crit_reduced = stats.f.ppf(1 - alpha, df_reg_reduced, df_res_reduced)

    print(f"\n--- Повторная проверка адекватности ---")
    print(f"Степени свободы регрессии: df_reg = {df_reg_reduced}")
    print(f"Степени свободы остатков: df_res = {df_res_reduced}")
    print(f"F-статистика: F = {F_stat_reduced:.6f}")
    print(f"Критическое значение F_{{{alpha}, {df_reg_reduced}, {df_res_reduced}}} = {F_crit_reduced:.6f}")

    print("\nВывод:")
    if F_stat_reduced > F_crit_reduced:
        print(f"F ({F_stat_reduced:.4f}) > F_crit ({F_crit_reduced:.4f})")
        print(f"Упрощенное уравнение регрессии АДЕКВАТНО экспериментальным данным (α = {alpha})")
    else:
        print(f"F ({F_stat_reduced:.4f}) <= F_crit ({F_crit_reduced:.4f})")
        print(f"Упрощенное уравнение регрессии НЕ АДЕКВАТНО экспериментальным данным (α = {alpha})")

    # Сохраняем итоговую модель для графиков/итогов
    final_indices = significant_indices
    b_final = b_reduced
    X_final = X_reduced
    y_pred_final = y_pred_reduced
    residuals_final = residuals_reduced
    SS_res_final = SS_res_reduced
    SS_reg_final = SS_reg_reduced
    R2_final = R2_reduced
    df_reg_final = df_reg_reduced
    df_res_final = df_res_reduced
    F_stat_final = F_stat_reduced
    F_crit_final = F_crit_reduced
else:
    print("\nВсе коэффициенты (кроме возможного b₀) значимы, упрощение модели не требуется.")

# ========== ШАГ 8: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ==========

print("ШАГ 8: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

# Создаем фигуру с несколькими графиками
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Результаты регрессионного анализа (Вариант 7)', fontsize=14, fontweight='bold')

# График 1: Исходные данные и кривая регрессии
ax1 = axes[0, 0]
x_plot = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)

# Кривая ИТОГОВОЙ модели (полная или упрощенная)
# Собираем X для x_plot только по тем столбцам, которые остались в final_indices
X_plot = np.column_stack([np.ones_like(x_plot), x_plot, x_plot**2])[:, final_indices]
y_plot = X_plot @ b_final

# подпись модели
labels_all = ['1', 'x', 'x²']
terms = []
for coef, idx in zip(b_final, final_indices):
    if idx == 0:
        terms.append(f"{coef:.2f}")
    else:
        terms.append(f"{coef:.2f}{labels_all[idx]}")
eq_label = " + ".join(terms).replace("+ -", "- ")

ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'Регрессия: y={eq_label}')
ax1.scatter(x, y_pred_final, color='blue', s=50, marker='x', zorder=4, label='Расчетные значения')

# Вертикальные линии для остатков
for i in range(n):
    ax1.plot([x[i], x[i]], [y[i], y_pred_final[i]], 'g--', alpha=0.5, linewidth=1)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('График регрессии')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Остатки
ax2 = axes[0, 1]
ax2.scatter(x, residuals_final, color='green', s=100, edgecolors='black')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel('Остатки')
ax2.set_title('График остатков')
ax2.grid(True, alpha=0.3)

# График 3: Q-Q plot для проверки нормальности остатков
ax3 = axes[1, 0]
stats.probplot(residuals_final, dist="norm", plot=ax3)
ax3.set_title('Q-Q график (проверка нормальности остатков)')
ax3.grid(True, alpha=0.3)

# График 4: Гистограмма остатков
ax4 = axes[1, 1]
ax4.hist(residuals_final, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Остатки')
ax4.set_ylabel('Частота')
ax4.set_title('Гистограмма остатков')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('regression_analysis_results.png', dpi=300, bbox_inches='tight')
print("\nГрафики сохранены в файл 'regression_analysis_results.png'")

# ========== ИТОГОВЫЙ ВЫВОД ==========

print("ИТОГОВЫЙ ВЫВОД")

# печать уравнения итоговой модели
model_terms = []
for coef, idx in zip(b_final, final_indices):
    if idx == 0:
        model_terms.append(f"{coef:.6f}")
    elif idx == 1:
        model_terms.append(f"{coef:.6f}·x")
    elif idx == 2:
        model_terms.append(f"{coef:.6f}·x²")

print(f"\n1. Построено уравнение регрессии: y = {' + '.join(model_terms).replace('+ -', '- ')}")

print(f"\n2. Коэффициент детерминации R² = {R2_final:.6f}")
print(f"   Модель объясняет {R2_final*100:.2f}% вариации данных")

print(f"\n3. Проверка адекватности (F-тест, α = {alpha}):")
print(f"   F = {F_stat_final:.4f}, F_crit = {F_crit_final:.4f}")
if F_stat_final > F_crit_final:
    print("   Модель АДЕКВАТНА")
else:
    print("   Модель НЕ АДЕКВАТНА")

print(f"\n4. Проверка значимости коэффициентов (t-тест, α = {alpha}):")
for i, label in enumerate(['b₀', 'b₁', 'b₂']):
    print(f"   {label}: t = {t_stats[i]:.4f}, значим = {'ДА' if significant[i] else 'НЕТ'}")

if insignificant_indices:
    print(f"\n5. Упрощенная модель после исключения незначимых коэффициентов:")
    model_terms = []
    for coef, idx in zip(b_final, final_indices):
        if idx == 0:
            model_terms.append(f"{coef:.6f}")
        elif idx == 1:
            model_terms.append(f"{coef:.6f}·x")
        elif idx == 2:
            model_terms.append(f"{coef:.6f}·x²")
    print(f"   y = {' + '.join(model_terms).replace('+ -', '- ')}")
else:
    print(f"\n5. Упрощение не требуется (b₀ оставляем, а b₁ и b₂ значимы).")

plt.show()
