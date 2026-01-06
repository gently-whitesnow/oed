#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №5. Вариант 7. Часть 1
Многофакторный регрессионный анализ

Задача: На основе заданного массива данных построить уравнение регрессии
в виде линейного алгебраического полинома от двух переменных,
проверить его адекватность и значимость факторов.
Расчёты произвести в матричной форме.
"""

import numpy as np
from scipy import stats


# 1. ИСХОДНЫЕ ДАННЫЕ
# Исходные экспериментальные данные (вариант 7, таблица 5.1)
x1_original = np.array([1, 0.5, 3, 2, 1, -2])
x2_original = np.array([1, 2, 2, 3, 0.3, 0.5])
y_original = np.array([2, 4.3, 7.2, 8, 0, -3])

n = len(y_original)  # Количество экспериментов

print(f"\n{'Исходные экспериментальные данные:'}")
print(f"{'Количество экспериментов: n = '}{n}")
print(f"\n{'№':>3} {'x1':>8} {'x2':>8} {'y':>8}")
print("-"*32)
for i in range(n):
    print(f"{i+1:3d} {x1_original[i]:8.1f} {x2_original[i]:8.1f} {y_original[i]:8.1f}")

# 2. ЦЕНТРИРОВАНИЕ ФАКТОРОВ

print("ШАГ 1: ЦЕНТРИРОВАНИЕ ФАКТОРОВ")

# Вычисляем средние значения факторов
x1_mean = np.mean(x1_original)
x2_mean = np.mean(x2_original)
y_mean = np.mean(y_original)

print(f"\nСредние значения:")
print(f"  x̄₁ = {x1_mean:.4f}")
print(f"  x̄₂ = {x2_mean:.4f}")
print(f"  ȳ  = {y_mean:.4f}")

# Центрируем факторы (вычитаем средние значения)
x1_centered = x1_original - x1_mean
x2_centered = x2_original - x2_mean
y_centered = y_original - y_mean

print(f"\nЦентрированные данные:")
print(f"{'№':>3} {'x1-x̄₁':>10} {'x2-x̄₂':>10} {'y-ȳ':>10}")
print("-"*36)
for i in range(n):
    print(f"{i+1:3d} {x1_centered[i]:10.4f} {x2_centered[i]:10.4f} {y_centered[i]:10.4f}")

# 3. СОСТАВЛЕНИЕ МАТРИЧНОГО УРАВНЕНИЯ

print("ШАГ 2: СОСТАВЛЕНИЕ МАТРИЧНОГО УРАВНЕНИЯ")

# Для центрированных данных уравнение регрессии имеет вид:
# y - ȳ = b₁(x₁ - x̄₁) + b₂(x₂ - x̄₂)
#
# В матричной форме: (X'X)b = X'Y
# где X - матрица центрированных факторов (без столбца единиц!)
#     Y - вектор центрированных значений y
#     b - вектор коэффициентов регрессии [b₁, b₂]

# Формируем матрицу X (центрированные факторы)
X = np.column_stack([x1_centered, x2_centered])
Y = y_centered

print(f"\nМатрица центрированных факторов X ({n}×2):")
print(X)

print(f"\nВектор центрированных значений Y ({n}×1):")
print(Y)

# Вычисляем X'X (матрица Грама)
XTX = X.T @ X

print(f"\nМатрица X'X (2×2):")
print(XTX)

# Вычисляем X'Y
XTY = X.T @ Y

print(f"\nВектор X'Y (2×1):")
print(XTY)

# 4. РЕШЕНИЕ МАТРИЧНОГО УРАВНЕНИЯ

print("ШАГ 3: РЕШЕНИЕ МАТРИЧНОГО УРАВНЕНИЯ")

# Решаем систему (X'X)b = X'Y
# b = (X'X)⁻¹ X'Y
b_coeffs = np.linalg.solve(XTX, XTY)

print(f"\nКоэффициенты регрессии для центрированных переменных:")
print(f"  b₁ = {b_coeffs[0]:.6f}")
print(f"  b₂ = {b_coeffs[1]:.6f}")

# Для исходных (нецентрированных) переменных нужно найти b₀:
# b₀ = ȳ - b₁·x̄₁ - b₂·x̄₂
b0 = y_mean - b_coeffs[0] * x1_mean - b_coeffs[1] * x2_mean

print(f"\nСвободный член для исходных переменных:")
print(f"  b₀ = ȳ - b₁·x̄₁ - b₂·x̄₂")
print(f"  b₀ = {y_mean:.4f} - {b_coeffs[0]:.6f}·{x1_mean:.4f} - {b_coeffs[1]:.6f}·{x2_mean:.4f}")
print(f"  b₀ = {b0:.6f}")

print(f"\n{'УРАВНЕНИЕ РЕГРЕССИИ:'}")
print(f"{'Для центрированных переменных:'}")
print(f"  (y - {y_mean:.4f}) = {b_coeffs[0]:.6f}·(x₁ - {x1_mean:.4f}) + {b_coeffs[1]:.6f}·(x₂ - {x2_mean:.4f})")
print(f"\n{'Для исходных переменных:'}")
print(f"  ŷ = {b0:.6f} + {b_coeffs[0]:.6f}·x₁ + {b_coeffs[1]:.6f}·x₂")

# Вычисляем предсказанные значения
y_predicted = b0 + b_coeffs[0] * x1_original + b_coeffs[1] * x2_original

# Вычисляем остатки (ошибки)
residuals = y_original - y_predicted

print(f"\n{'Таблица расчётных значений и остатков:'}")
print(f"{'№':>3} {'x1':>8} {'x2':>8} {'y факт':>10} {'ŷ расч':>10} {'Остаток':>10}")
print("-"*53)
for i in range(n):
    print(f"{i+1:3d} {x1_original[i]:8.1f} {x2_original[i]:8.1f} {y_original[i]:10.4f} {y_predicted[i]:10.4f} {residuals[i]:10.4f}")

# 5. ПРОВЕРКА АДЕКВАТНОСТИ ПО КРИТЕРИЮ ФИШЕРА
print("ШАГ 4: ПРОВЕРКА АДЕКВАТНОСТИ ПО КРИТЕРИЮ ФИШЕРА (α = 0.05)")

# Количество коэффициентов (включая b₀)
p = 3  # b₀, b₁, b₂

# Общая сумма квадратов (Total Sum of Squares)
SST = np.sum((y_original - y_mean)**2)

# Регрессионная сумма квадратов (Regression Sum of Squares)
SSR = np.sum((y_predicted - y_mean)**2)

# Остаточная сумма квадратов (Residual Sum of Squares)
SSE = np.sum(residuals**2)

print(f"\nСуммы квадратов:")
print(f"  SST (общая)       = {SST:.6f}")
print(f"  SSR (регрессия)   = {SSR:.6f}")
print(f"  SSE (остаточная)  = {SSE:.6f}")
print(f"  Проверка: SST = SSR + SSE")
print(f"           {SST:.6f} = {SSR:.6f} + {SSE:.6f} = {SSR + SSE:.6f}")

# Степени свободы
df_regression = p - 1  # количество факторов (без b₀)
df_residual = n - p    # n - количество коэффициентов
df_total = n - 1

print(f"\nСтепени свободы:")
print(f"  df_регрессия = p - 1 = {df_regression}")
print(f"  df_остаток   = n - p = {df_residual}")
print(f"  df_общая     = n - 1 = {df_total}")

# Средние квадраты
MSR = SSR / df_regression  # Mean Square Regression
MSE = SSE / df_residual    # Mean Square Error

print(f"\nСредние квадраты:")
print(f"  MSR = SSR / df_регрессия = {SSR:.6f} / {df_regression} = {MSR:.6f}")
print(f"  MSE = SSE / df_остаток   = {SSE:.6f} / {df_residual} = {MSE:.6f}")

# F-статистика
F_statistic = MSR / MSE

print(f"\nF-статистика:")
print(f"  F = MSR / MSE = {MSR:.6f} / {MSE:.6f} = {F_statistic:.6f}")

# Критическое значение F-распределения
alpha = 0.05
F_critical = stats.f.ppf(1 - alpha, df_regression, df_residual)

print(f"\nКритическое значение F-распределения:")
print(f"  F_критич(α={alpha}, df1={df_regression}, df2={df_residual}) = {F_critical:.6f}")

# Проверка адекватности
print(f"\n{'ПРОВЕРКА АДЕКВАТНОСТИ:'}")
if F_statistic > F_critical:
    print(f"  F_расч ({F_statistic:.6f}) > F_критич ({F_critical:.6f})")
    print(f"  ✓ Уравнение регрессии АДЕКВАТНО экспериментальным данным")
    is_adequate = True
else:
    print(f"  F_расч ({F_statistic:.6f}) ≤ F_критич ({F_critical:.6f})")
    print(f"  ✗ Уравнение регрессии НЕ АДЕКВАТНО экспериментальным данным")
    is_adequate = False

# Коэффициент детерминации R²
R_squared = SSR / SST

print(f"\nКоэффициент детерминации:")
print(f"  R² = SSR / SST = {SSR:.6f} / {SST:.6f} = {R_squared:.6f}")
print(f"  Регрессия объясняет {R_squared*100:.2f}% вариации зависимой переменной")

# 6. СЕЛЕКЦИЯ ФАКТОРОВ ПО КРИТЕРИЮ СТЬЮДЕНТА

print("ШАГ 5: СЕЛЕКЦИЯ ФАКТОРОВ ПО КРИТЕРИЮ СТЬЮДЕНТА (α = 0.05)")

# Дисперсия остатков
s_squared = MSE

# Вычисляем ковариационную матрицу коэффициентов
# Для центрированных переменных: Cov(b) = s² · (X'X)⁻¹
XTX_inv = np.linalg.inv(XTX)
cov_matrix = s_squared * XTX_inv

# Стандартные ошибки коэффициентов b₁ и b₂
se_b1 = np.sqrt(cov_matrix[0, 0])
se_b2 = np.sqrt(cov_matrix[1, 1])

print(f"\nДисперсия остатков:")
print(f"  s² = MSE = {s_squared:.6f}")
print(f"  s = {np.sqrt(s_squared):.6f}")

print(f"\nМатрица (X'X)⁻¹:")
print(XTX_inv)

print(f"\nСтандартные ошибки коэффициентов:")
print(f"  SE(b₁) = {se_b1:.6f}")
print(f"  SE(b₂) = {se_b2:.6f}")

# t-статистики для коэффициентов
t_b1 = b_coeffs[0] / se_b1
t_b2 = b_coeffs[1] / se_b2

print(f"\nt-статистики:")
print(f"  t(b₁) = b₁ / SE(b₁) = {b_coeffs[0]:.6f} / {se_b1:.6f} = {t_b1:.6f}")
print(f"  t(b₂) = b₂ / SE(b₂) = {b_coeffs[1]:.6f} / {se_b2:.6f} = {t_b2:.6f}")

# Критическое значение t-распределения (двусторонний тест)
t_critical = stats.t.ppf(1 - alpha/2, df_residual)

print(f"\nКритическое значение t-распределения:")
print(f"  t_критич(α/2={alpha/2}, df={df_residual}) = {t_critical:.6f}")

# Проверка значимости факторов
print(f"\n{'ПРОВЕРКА ЗНАЧИМОСТИ ФАКТОРОВ:'}")

significant_factors = []

print(f"\nФактор x₁:")
if abs(t_b1) > t_critical:
    print(f"  |t(b₁)| = {abs(t_b1):.6f} > t_критич = {t_critical:.6f}")
    print(f"  ✓ Фактор x₁ ЗНАЧИМ")
    significant_factors.append('x1')
else:
    print(f"  |t(b₁)| = {abs(t_b1):.6f} ≤ t_критич = {t_critical:.6f}")
    print(f"  ✗ Фактор x₁ НЕ ЗНАЧИМ")

print(f"\nФактор x₂:")
if abs(t_b2) > t_critical:
    print(f"  |t(b₂)| = {abs(t_b2):.6f} > t_критич = {t_critical:.6f}")
    print(f"  ✓ Фактор x₂ ЗНАЧИМ")
    significant_factors.append('x2')
else:
    print(f"  |t(b₂)| = {abs(t_b2):.6f} ≤ t_критич = {t_critical:.6f}")
    print(f"  ✗ Фактор x₂ НЕ ЗНАЧИМ")

print(f"\n{'Итого:'} Количество значимых факторов: {len(significant_factors)}")
if len(significant_factors) > 0:
    print(f"  Значимые факторы: {', '.join(significant_factors)}")

# 7. ПОВТОРНАЯ ПРОВЕРКА АДЕКВАТНОСТИ (если есть незначимые факторы)

if len(significant_factors) < 2:
    
    print("ШАГ 6: ПОВТОРНАЯ ПРОВЕРКА АДЕКВАТНОСТИ ПОСЛЕ ИСКЛЮЧЕНИЯ НЕЗНАЧИМЫХ ФАКТОРОВ")
    
    if len(significant_factors) == 0:
        print("\n⚠ Все факторы оказались незначимыми!")
        print("  Регрессионная модель неприменима для данных.")
        print("  Рекомендуется:")
        print("  - Проверить качество экспериментальных данных")
        print("  - Рассмотреть другие факторы")
        print("  - Попробовать нелинейную модель")
    else:
        # Строим модель только со значимыми факторами
        print(f"\nПостроение модели только со значимыми факторами: {', '.join(significant_factors)}")

        if 'x1' in significant_factors and 'x2' not in significant_factors:
            # Только x1
            X_reduced = x1_centered.reshape(-1, 1)
            factor_names = ['x₁']
        elif 'x2' in significant_factors and 'x1' not in significant_factors:
            # Только x2
            X_reduced = x2_centered.reshape(-1, 1)
            factor_names = ['x₂']

        # Решаем редуцированное уравнение
        XTX_reduced = X_reduced.T @ X_reduced
        XTY_reduced = X_reduced.T @ Y
        b_reduced = np.linalg.solve(XTX_reduced, XTY_reduced)

        print(f"\nКоэффициенты регрессии (редуцированная модель):")
        for i, name in enumerate(factor_names):
            print(f"  b_{name} = {b_reduced[i]:.6f}")

        # Вычисляем b₀ для редуцированной модели
        if 'x1' in significant_factors and 'x2' not in significant_factors:
            b0_reduced = y_mean - b_reduced[0] * x1_mean
            y_pred_reduced = b0_reduced + b_reduced[0] * x1_original
            print(f"\nУравнение регрессии (редуцированная модель):")
            print(f"  ŷ = {b0_reduced:.6f} + {b_reduced[0]:.6f}·x₁")
        else:
            b0_reduced = y_mean - b_reduced[0] * x2_mean
            y_pred_reduced = b0_reduced + b_reduced[0] * x2_original
            print(f"\nУравнение регрессии (редуцированная модель):")
            print(f"  ŷ = {b0_reduced:.6f} + {b_reduced[0]:.6f}·x₂")

        # Проверка адекватности редуцированной модели
        residuals_reduced = y_original - y_pred_reduced
        SSE_reduced = np.sum(residuals_reduced**2)
        SSR_reduced = np.sum((y_pred_reduced - y_mean)**2)

        p_reduced = 2  # b₀ и один коэффициент
        df_reg_reduced = p_reduced - 1
        df_res_reduced = n - p_reduced

        MSR_reduced = SSR_reduced / df_reg_reduced
        MSE_reduced = SSE_reduced / df_res_reduced
        F_reduced = MSR_reduced / MSE_reduced
        F_crit_reduced = stats.f.ppf(1 - alpha, df_reg_reduced, df_res_reduced)

        print(f"\nПроверка адекватности редуцированной модели:")
        print(f"  SSR = {SSR_reduced:.6f}")
        print(f"  SSE = {SSE_reduced:.6f}")
        print(f"  MSR = {MSR_reduced:.6f}")
        print(f"  MSE = {MSE_reduced:.6f}")
        print(f"  F_расч = {F_reduced:.6f}")
        print(f"  F_критич(α={alpha}, df1={df_reg_reduced}, df2={df_res_reduced}) = {F_crit_reduced:.6f}")

        if F_reduced > F_crit_reduced:
            print(f"\n  ✓ Редуцированная модель АДЕКВАТНА")
        else:
            print(f"\n  ✗ Редуцированная модель НЕ АДЕКВАТНА")

        R_squared_reduced = SSR_reduced / SST
        print(f"\n  R² = {R_squared_reduced:.6f} ({R_squared_reduced*100:.2f}%)")
else:
    
    print("ШАГ 6: ПОВТОРНАЯ ПРОВЕРКА АДЕКВАТНОСТИ")
    
    print("\n✓ Все факторы значимы, исключение факторов не требуется.")
    print("  Построенная модель полна и адекватна.")


# ИТОГОВЫЕ РЕЗУЛЬТАТЫ

print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")

print(f"\nПолное уравнение регрессии:")
print(f"  ŷ = {b0:.6f} + {b_coeffs[0]:.6f}·x₁ + {b_coeffs[1]:.6f}·x₂")

print(f"\nСтатистические характеристики:")
print(f"  R² = {R_squared:.6f} ({R_squared*100:.2f}%)")
print(f"  F-статистика = {F_statistic:.6f}")
print(f"  Адекватность: {'ДА' if is_adequate else 'НЕТ'}")

print(f"\nЗначимость факторов (α = {alpha}):")
print(f"  x₁: {'ЗНАЧИМ' if 'x1' in significant_factors else 'НЕ ЗНАЧИМ'} (|t| = {abs(t_b1):.4f}, t_крит = {t_critical:.4f})")
print(f"  x₂: {'ЗНАЧИМ' if 'x2' in significant_factors else 'НЕ ЗНАЧИМ'} (|t| = {abs(t_b2):.4f}, t_крит = {t_critical:.4f})")

