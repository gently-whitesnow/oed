#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №2. Вариант 7
Выравнивание статистических распределений и проверка гипотез о законах распределения случайных величин

Задача: По заданному интервальному статистическому ряду построить статистическое распределение
экспериментальных данных в виде гистограммы, произвести ее выравнивание теоретической плотностью
нормального распределения и проверить гипотезу о соответствии статистического и теоретического распределений.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Настройка поддержки русского языка в графиках
plt.rcParams['font.family'] = 'DejaVu Sans'


def main():
    """Главная функция программы"""
    # ИСХОДНЫЕ ДАННЫЕ ДЛЯ ВАРИАНТА 7

    # Интервалы Jl
    intervals = [
        (-0.5, 0),
        (0, 0.5),
        (0.5, 1),
        (1, 1.5),
        (1.5, 2),
        (2, 2.5),
        (2.5, 3)
    ]

    # Число попаданий ml в каждый интервал
    frequencies = np.array([2, 10, 29, 30, 21, 7, 1])

    # Уровень значимости (для нечетных вариантов)
    alpha = 0.05

    print("\nИСХОДНЫЕ ДАННЫЕ:")
    
    print(f"{'Интервал Jl':<20} {'Число попаданий ml':<20}")
    
    for interval, freq in zip(intervals, frequencies):
        print(f"[{interval[0]:.1f}; {interval[1]:.1f}){' '*10} {freq}")
    
    print(f"Уровень значимости α = {alpha}")
    print()

    # ШАГ 1: ВЫЧИСЛЕНИЕ СТАТИСТИЧЕСКИХ ВЕРОЯТНОСТЕЙ
    
    print("ШАГ 1: ВЫЧИСЛЕНИЕ СТАТИСТИЧЕСКИХ ВЕРОЯТНОСТЕЙ")
    
    # Общее число наблюдений
    n_total = np.sum(frequencies)
    print(f"Общее число наблюдений n = {n_total}")

    # Статистические вероятности
    probabilities = frequencies / n_total

    print("\nСтатистические вероятности попадания в интервалы:")
    
    print(f"{'Интервал Jl':<20} {'ml':<10} {'p*l = ml/n':<20}")
    
    for interval, freq, prob in zip(intervals, frequencies, probabilities):
        print(f"[{interval[0]:.1f}; {interval[1]:.1f}){' '*10} {freq:<10} {prob:.6f}")
    
    print(f"Сумма вероятностей: {np.sum(probabilities):.6f} (должна быть ≈ 1)")

    # Вычисление середин интервалов
    interval_centers = np.array([(a + b) / 2 for a, b in intervals])
    print(f"\nСередины интервалов xl: {interval_centers}")

    # Ширина интервалов
    interval_widths = np.array([b - a for a, b in intervals])
    print(f"Ширина интервалов Δl: {interval_widths}")
    # ШАГ 2: ПОСТРОЕНИЕ ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЯ

    print("ШАГ 2: ПОСТРОЕНИЕ ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЯ")
    
    # Вычисление плотности для гистограммы
    # Плотность = частота / (общее число * ширина интервала)
    histogram_density = frequencies / (n_total * interval_widths)

    print("\nПлотность распределения для гистограммы:")
    
    print(f"{'Интервал Jl':<20} {'p*l/Δl':<20}")
    
    for interval, density in zip(intervals, histogram_density):
        print(f"[{interval[0]:.1f}; {interval[1]:.1f}){' '*10} {density:.6f}")
    
    # ШАГ 3: НАХОЖДЕНИЕ ТЕОРЕТИЧЕСКОЙ ПЛОТНОСТИ НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ
    # МЕТОДОМ МОМЕНТОВ

    print("ШАГ 3: НАХОЖДЕНИЕ ТЕОРЕТИЧЕСКОЙ ПЛОТНОСТИ НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ")
    
    # Метод моментов: оценки параметров нормального распределения
    # Выборочное среднее (математическое ожидание)
    mean_estimate = np.sum(interval_centers * frequencies) / n_total

    # Выборочная дисперсия
    variance_estimate = np.sum(((interval_centers - mean_estimate) ** 2) * frequencies) / n_total

    # Среднеквадратическое отклонение
    std_estimate = np.sqrt(variance_estimate)

    # Исправленная (несмещенная) оценка СКО
    std_corrected = np.sqrt(n_total / (n_total - 1)) * std_estimate

    print(f"\nОценка математического ожидания (μ̂): {mean_estimate:.4f}")
    print(f"Оценка дисперсии (σ̂²): {variance_estimate:.4f}")
    print(f"Оценка СКО (σ̂): {std_estimate:.4f}")
    print(f"Исправленная оценка СКО (s): {std_corrected:.4f}")

    # Используем исправленную оценку для построения теоретической плотности
    print(f"\nПараметры нормального распределения N(μ, σ²):")
    print(f"  μ = {mean_estimate:.4f}")
    print(f"  σ = {std_corrected:.4f}")

    # Вычисление теоретической плотности в центрах интервалов
    theoretical_density = stats.norm.pdf(interval_centers, mean_estimate, std_corrected)

    print("\nТеоретическая плотность в центрах интервалов:")
    
    print(f"{'xl':<10} {'f(xl)':<20}")
    
    for center, density in zip(interval_centers, theoretical_density):
        print(f"{center:<10.2f} {density:.6f}")
    
    # ШАГ 4: ПРОВЕРКА ГИПОТЕЗЫ МЕТОДОМ К. ПИРСОНА (χ²-критерий)

    print("ШАГ 4: ПРОВЕРКА ГИПОТЕЗЫ МЕТОДОМ К. ПИРСОНА")
    
    # Вычисление теоретических вероятностей для каждого интервала
    # P(Jl) = Φ((bl - μ)/σ) - Φ((al - μ)/σ)
    theoretical_probs = []
    for interval in intervals:
        a, b = interval
        prob = stats.norm.cdf(b, mean_estimate, std_corrected) - \
               stats.norm.cdf(a, mean_estimate, std_corrected)
        theoretical_probs.append(prob)

    theoretical_probs = np.array(theoretical_probs)

    # Теоретические частоты
    theoretical_frequencies = n_total * theoretical_probs

    print("\nТеоретические вероятности и частоты:")
    
    print(f"{'Интервал Jl':<20} {'P(Jl)':<15} {'n·P(Jl)':<15} {'ml':<10}")
    
    for interval, prob, theor_freq, obs_freq in zip(intervals, theoretical_probs,
                                                      theoretical_frequencies, frequencies):
        print(f"[{interval[0]:.1f}; {interval[1]:.1f}){' '*10} {prob:<15.6f} {theor_freq:<15.2f} {obs_freq}")
    

    # Объединение интервалов с малыми теоретическими частотами (< 5)
    print("\nПроверка условия применимости критерия Пирсона (n·P(Jl) ≥ 5):")

    # Найдем интервалы, которые нужно объединить
    grouped_intervals = []
    grouped_observed = []
    grouped_theoretical = []

    i = 0
    while i < len(intervals):
        if theoretical_frequencies[i] < 5:
            # Объединяем с соседними интервалами
            combined_interval = [intervals[i][0], intervals[i][1]]
            combined_obs = frequencies[i]
            combined_theor = theoretical_frequencies[i]
            j = i + 1

            # Объединяем следующие интервалы, пока не достигнем частоты ≥ 5
            while j < len(intervals) and combined_theor < 5:
                combined_interval[1] = intervals[j][1]
                combined_obs += frequencies[j]
                combined_theor += theoretical_frequencies[j]
                j += 1

            grouped_intervals.append(tuple(combined_interval))
            grouped_observed.append(combined_obs)
            grouped_theoretical.append(combined_theor)
            i = j
        else:
            grouped_intervals.append(intervals[i])
            grouped_observed.append(frequencies[i])
            grouped_theoretical.append(theoretical_frequencies[i])
            i += 1

    while grouped_theoretical[-1] < 5 and len(grouped_theoretical) > 1:
        # объединяем последний интервал с предыдущим
        grouped_intervals[-2] = (grouped_intervals[-2][0], grouped_intervals[-1][1])
        grouped_observed[-2] += grouped_observed[-1]
        grouped_theoretical[-2] += grouped_theoretical[-1]
        grouped_intervals.pop()
        grouped_observed = grouped_observed[:-1]
        grouped_theoretical = grouped_theoretical[:-1]

    grouped_observed = np.array(grouped_observed)
    grouped_theoretical = np.array(grouped_theoretical)

    if len(grouped_intervals) < len(intervals):
        print("Необходимо объединение интервалов!")
        print("\nОбъединенные интервалы:")
        
        print(f"{'Интервал':<20} {'ml (набл.)':<15} {'n·P(Jl) (теор.)':<20}")
        
        for interval, obs, theor in zip(grouped_intervals, grouped_observed, grouped_theoretical):
            print(f"[{interval[0]:.1f}; {interval[1]:.1f}){' '*10} {obs:<15} {theor:<20.2f}")
        
    else:
        print("Все теоретические частоты ≥ 5, объединение не требуется.")

    # Вычисление статистики χ²
    chi_squared_stat = np.sum((grouped_observed - grouped_theoretical) ** 2 / grouped_theoretical)

    # Число степеней свободы
    # k = число интервалов - 1 - число оцененных параметров (2 для нормального распределения: μ и σ)
    k = len(grouped_intervals) - 1 - 2

    print(f"\nВычисление статистики χ²:")
    
    print(f"{'Интервал':<20} {'ml':<10} {'n·P(Jl)':<15} {'(ml - n·P)²/n·P':<20}")
    
    for interval, obs, theor in zip(grouped_intervals, grouped_observed, grouped_theoretical):
        chi_component = (obs - theor) ** 2 / theor
        print(f"[{interval[0]:.1f}; {interval[1]:.1f}){' '*10} {obs:<10} {theor:<15.2f} {chi_component:<20.6f}")
    
    print(f"Наблюдаемое значение статистики χ²набл = {chi_squared_stat:.6f}")

    # Критическое значение χ² для уровня значимости α
    chi_squared_critical = stats.chi2.ppf(1 - alpha, k)

    print(f"\nЧисло степеней свободы k = {k}")
    print(f"Уровень значимости α = {alpha}")
    print(f"Критическое значение χ²крит({k}, {alpha}) = {chi_squared_critical:.6f}")

    # Проверка гипотезы
    
    print("РЕЗУЛЬТАТ ПРОВЕРКИ ГИПОТЕЗЫ")
    
    print(f"\nНаблюдаемое значение: χ²набл = {chi_squared_stat:.6f}")
    print(f"Критическое значение: χ²крит = {chi_squared_critical:.6f}")

    if chi_squared_stat < chi_squared_critical:
        print(f"\n✓ χ²набл < χ²крит ({chi_squared_stat:.6f} < {chi_squared_critical:.6f})")
    else:
        print(f"\n✗ χ²набл ≥ χ²крит ({chi_squared_stat:.6f} ≥ {chi_squared_critical:.6f})")

    # Вычисление p-value
    p_value = 1 - stats.chi2.cdf(chi_squared_stat, k)
    print(f"\np-value = {p_value:.6f}")

    # ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('Выравнивание статистических распределений',
             fontsize=14, fontweight='bold')

    # График 1: Гистограмма и теоретическая плотность
    ax1, ax2, ax3 = axes

    # Построение гистограммы
    for i, (interval, density) in enumerate(zip(intervals, histogram_density)):
        ax1.bar(interval_centers[i], density, width=interval_widths[i],
                alpha=0.6, edgecolor='black', color='skyblue', label='Гистограмма' if i == 0 else '')

    # Построение теоретической плотности (непрерывная кривая)
    x_range = np.linspace(intervals[0][0] - 0.5, intervals[-1][1] + 0.5, 1000)
    y_theory = stats.norm.pdf(x_range, mean_estimate, std_corrected)
    ax1.plot(x_range, y_theory, 'r-', linewidth=2, label='Теоретическая плотность N(μ,σ²)')

    ax1.set_xlabel('Значение случайной величины')
    ax1.set_ylabel('Плотность распределения')
    ax1.set_title('Гистограмма и теоретическая плотность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Сравнение наблюдаемых и теоретических частот
    x_pos = np.arange(len(intervals))
    width = 0.35

    ax2.bar(x_pos - width/2, frequencies, width, label='Наблюдаемые ml',
            alpha=0.7, edgecolor='black', color='skyblue')
    ax2.bar(x_pos + width/2, theoretical_frequencies, width, label='Теоретические n·P(Jl)',
            alpha=0.7, edgecolor='black', color='salmon')

    interval_labels = [f"[{a:.1f};{b:.1f})" for a, b in intervals]
    ax2.set_xlabel('Интервалы')
    ax2.set_ylabel('Частота')
    ax2.set_title('Сравнение наблюдаемых и теоретических частот')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(interval_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # График 3: Эмпирическая и теоретическая функции распределения

    # Эмпирическая функция распределения
    cumulative_probs = np.cumsum(probabilities)
    cumulative_probs = np.insert(cumulative_probs, 0, 0)
    interval_bounds = [intervals[0][0]] + [b for a, b in intervals]

    # Построение ступенчатой эмпирической функции
    for i in range(len(intervals)):
        ax3.hlines(cumulative_probs[i], interval_bounds[i], interval_bounds[i+1],
                   colors='blue', linewidth=2, label='Эмпирическая F*(x)' if i == 0 else '')
        if i < len(intervals) - 1:
            ax3.vlines(interval_bounds[i+1], cumulative_probs[i], cumulative_probs[i+1],
                       colors='blue', linewidth=2, linestyles='dashed', alpha=0.5)

    # Теоретическая функция распределения
    x_range = np.linspace(intervals[0][0] - 0.5, intervals[-1][1] + 0.5, 1000)
    y_cdf = stats.norm.cdf(x_range, mean_estimate, std_corrected)
    ax3.plot(x_range, y_cdf, 'r-', linewidth=2, label='Теоретическая Φ(x)')

    ax3.set_xlabel('Значение случайной величины')
    ax3.set_ylabel('Вероятность')
    ax3.set_title('Эмпирическая и теоретическая функции распределения')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # Сохранение графика
    plt.savefig('lab2_results.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
