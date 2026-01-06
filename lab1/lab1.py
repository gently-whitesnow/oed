#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №1. Вариант 7
Статистическое оценивание числовых характеристик законов распределения случайных величин

Задача: На основе массива экспериментальных данных найти оценку математического ожидания,
проверить качество оценивания по заданной доверительной вероятности и погрешности.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Настройка поддержки русского языка в графиках
plt.rcParams['font.family'] = 'DejaVu Sans'


class StatisticalAnalysis:
    """Класс для статистического анализа экспериментальных данных"""

    def __init__(self, data, beta, epsilon_beta):
        """
        Инициализация класса

        Параметры:
        ----------
        data : array_like
            Массив экспериментальных данных
        beta : float
            Заданная доверительная вероятность
        epsilon_beta : float
            Заданная максимальная вероятная погрешность
        """
        self.original_data = np.array(data)
        self.data = np.array(data)
        self.beta = beta
        self.epsilon_beta = epsilon_beta
        self.n = len(data)

    def calculate_mean(self, data=None):
        """
        Вычисление оценки математического ожидания (выборочное среднее)

        Параметры:
        ----------
        data : array_like, optional
            Данные для вычисления. Если None, используются текущие данные

        Возвращает:
        -----------
        float
            Оценка математического ожидания
        """
        if data is None:
            data = self.data
        return np.mean(data)

    def calculate_std(self, data=None):
        """
        Вычисление несмещенной оценки среднеквадратичного отклонения

        Параметры:
        ----------
        data : array_like, optional
            Данные для вычисления. Если None, используются текущие данные

        Возвращает:
        -----------
        float
            Оценка СКО
        """
        if data is None:
            data = self.data
        # ddof=1 для несмещенной оценки
        return np.std(data, ddof=1)

    def build_confidence_interval_95(self):
        """
        Построение 95% прогнозного интервала для случайной величины
        (используется для отсеивания аномальных наблюдений)

        Возвращает:
        -----------
        tuple
            (нижняя_граница, верхняя_граница)
        """
        mean = self.calculate_mean()
        std = self.calculate_std()
        n = len(self.data)

        # Квантиль распределения Стьюдента для 95% доверительного интервала
        alpha = 0.05
        t_quantile = stats.t.ppf(1 - alpha/2, n - 1)

        # Прогнозный интервал для отдельного наблюдения
        # Учитывает как неопределенность оценки среднего (1/n), так и вариабельность данных (1)
        margin = t_quantile * std * np.sqrt(1 + 1/n)

        lower_bound = mean - margin
        upper_bound = mean + margin

        return lower_bound, upper_bound

    def filter_outliers(self):
        """
        Отсеивание аномальных наблюдений, не попадающих в 95% доверительный интервал

        Возвращает:
        -----------
        tuple
            (отфильтрованные_данные, индексы_выбросов, нижняя_граница, верхняя_граница)
        """
        lower_bound, upper_bound = self.build_confidence_interval_95()

        # Поиск индексов выбросов
        outliers_indices = np.where((self.data < lower_bound) | (self.data > upper_bound))[0]

        # Фильтрация данных
        filtered_data = self.data[(self.data >= lower_bound) & (self.data <= upper_bound)]

        return filtered_data, outliers_indices, lower_bound, upper_bound

    def build_confidence_interval_mean(self, data, confidence_level):
        """
        Построение доверительного интервала для математического ожидания

        Параметры:
        ----------
        data : array_like
            Данные для анализа
        confidence_level : float
            Доверительная вероятность (например, 0.89)

        Возвращает:
        -----------
        tuple
            (нижняя_граница, верхняя_граница, полуширина_интервала)
        """
        mean = self.calculate_mean(data)
        std = self.calculate_std(data)
        n = len(data)

        # Квантиль распределения Стьюдента
        alpha = 1 - confidence_level
        t_quantile = stats.t.ppf(1 - alpha/2, n - 1)

        # Полуширина доверительного интервала
        margin = t_quantile * std / np.sqrt(n)

        lower_bound = mean - margin
        upper_bound = mean + margin

        return lower_bound, upper_bound, margin

    def calculate_confidence_probability(self, data, epsilon):
        """
        Вычисление доверительной вероятности для заданной погрешности

        Параметры:
        ----------
        data : array_like
            Данные для анализа
        epsilon : float
            Заданная максимальная вероятная погрешность

        Возвращает:
        -----------
        float
            Доверительная вероятность
        """
        std = self.calculate_std(data)
        n = len(data)

        # Вычисление значения t-статистики для заданной погрешности
        t_value = epsilon * np.sqrt(n) / std

        # Вычисление доверительной вероятности
        # P(-t < T < t) = 2 * F(t) - 1, где F - функция распределения Стьюдента
        confidence_prob = 2 * stats.t.cdf(t_value, n - 1) - 1

        return confidence_prob

    def print_results(self):
        """Вывод всех результатов анализа"""

        # 1. Исходные данные
        print("\n1. ИСХОДНЫЕ ДАННЫЕ")
        
        print(f"Массив экспериментальных данных (n={self.n}):")
        print(self.original_data)
        print(f"\nЗаданная доверительная вероятность β = {self.beta}")
        print(f"Заданная максимальная вероятная погрешность εβ = {self.epsilon_beta}")

        # 2. Первоначальная оценка математического ожидания
        print("\n2. ПЕРВОНАЧАЛЬНАЯ ОЦЕНКА МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ")
        
        initial_mean = self.calculate_mean()
        initial_std = self.calculate_std()
        print(f"Оценка математического ожидания: x̄ = {initial_mean:.4f}")
        print(f"Оценка СКО: s = {initial_std:.4f}")

        # 3. Построение 95% доверительного интервала
        print("\n3. ПОСТРОЕНИЕ 95% ДОВЕРИТЕЛЬНОГО ИНТЕРВАЛА")
        
        lower_95, upper_95 = self.build_confidence_interval_95()
        print(f"95% доверительный интервал для случайной величины:")
        print(f"[{lower_95:.4f}, {upper_95:.4f}]")

        # 4. Отсеивание аномальных наблюдений
        print("\n4. ОТСЕИВАНИЕ АНОМАЛЬНЫХ НАБЛЮДЕНИЙ")
        
        filtered_data, outliers_indices, lb, ub = self.filter_outliers()

        if len(outliers_indices) > 0:
            print(f"Обнаружено аномальных наблюдений: {len(outliers_indices)}")
            print(f"Индексы аномальных значений: {outliers_indices}")
            print(f"Аномальные значения: {self.original_data[outliers_indices]}")
        else:
            print("Аномальных наблюдений не обнаружено")

        print(f"\nДанные после отсеивания (n={len(filtered_data)}):")
        print(filtered_data)

        # 5. Уточненная оценка математического ожидания
        print("\n5. УТОЧНЕННАЯ ОЦЕНКА МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ")
        
        refined_mean = self.calculate_mean(filtered_data)
        refined_std = self.calculate_std(filtered_data)
        print(f"Уточненная оценка математического ожидания: x̄* = {refined_mean:.4f}")
        print(f"Уточненная оценка СКО: s* = {refined_std:.4f}")
        print(f"Изменение оценки: Δx̄ = {refined_mean - initial_mean:.4f}")

        # 6. Доверительный интервал для заданной доверительной вероятности
        print(f"\n6. ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ β = {self.beta}")
        
        lb_beta, ub_beta, margin_beta = self.build_confidence_interval_mean(
            filtered_data, self.beta
        )
        print(f"Доверительный интервал для математического ожидания:")
        print(f"[{lb_beta:.4f}, {ub_beta:.4f}]")
        print(f"Полуширина интервала: ε = {margin_beta:.4f}")
        print(f"С вероятностью {self.beta} математическое ожидание находится в указанном интервале")

        # 7. Доверительная вероятность для заданной погрешности
        print(f"\n7. ДОВЕРИТЕЛЬНАЯ ВЕРОЯТНОСТЬ ДЛЯ εβ = {self.epsilon_beta}")
        
        calc_beta = self.calculate_confidence_probability(filtered_data, self.epsilon_beta)
        print(f"Вычисленная доверительная вероятность: β = {calc_beta:.4f}")
        print(f"С вероятностью {calc_beta:.4f} математическое ожидание отклоняется")
        print(f"от оценки x̄* = {refined_mean:.4f} не более чем на {self.epsilon_beta}")

        # 8. Проверка качества оценивания
        print("\n8. ПРОВЕРКА КАЧЕСТВА ОЦЕНИВАНИЯ")
        
        print(f"Требуемая доверительная вероятность: β = {self.beta}")
        print(f"Полученная полуширина интервала: ε = {margin_beta:.4f}")
        print(f"Заданная максимальная погрешность: εβ = {self.epsilon_beta}")

        if margin_beta <= self.epsilon_beta:
            print(f"\n✓ Качество оценивания УДОВЛЕТВОРИТЕЛЬНОЕ:")
            print(f"  Полученная погрешность ({margin_beta:.4f}) не превышает заданную ({self.epsilon_beta})")
        else:
            print(f"\n✗ Качество оценивания НЕ УДОВЛЕТВОРИТЕЛЬНОЕ:")
            print(f"  Полученная погрешность ({margin_beta:.4f}) превышает заданную ({self.epsilon_beta})")
            print(f"  Для улучшения качества требуется увеличить объем выборки")


    def visualize_results(self):
        """Визуализация результатов анализа"""
        # Отсеивание выбросов
        filtered_data, outliers_indices, lb_95, ub_95 = self.filter_outliers()
        initial_mean = self.calculate_mean()
        refined_mean = self.calculate_mean(filtered_data)

        # Доверительный интервал для математического ожидания
        lb_beta, ub_beta, margin_beta = self.build_confidence_interval_mean(
            filtered_data, self.beta
        )

        # Создание фигуры с несколькими графиками
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Результаты статистического анализа (Вариант 7)',
                     fontsize=14, fontweight='bold')

        # График 1: Исходные данные с отмеченными выбросами
        ax1 = axes[0, 0]
        indices = np.arange(len(self.original_data))
        colors = ['red' if i in outliers_indices else 'blue'
                  for i in range(len(self.original_data))]
        ax1.scatter(indices, self.original_data, c=colors, s=100, alpha=0.6, edgecolors='black')
        ax1.axhline(y=lb_95, color='green', linestyle='--',
                    label=f'95% ПИ: [{lb_95:.2f}, {ub_95:.2f}]')
        ax1.axhline(y=ub_95, color='green', linestyle='--')
        ax1.axhline(y=initial_mean, color='orange', linestyle='-',
                    label=f'Среднее: {initial_mean:.2f}')
        ax1.set_xlabel('Номер наблюдения')
        ax1.set_ylabel('Значение')
        ax1.set_title('Исходные данные и выбросы')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Гистограмма до и после отсеивания
        ax2 = axes[0, 1]
        ax2.hist(self.original_data, bins=8, alpha=0.5, label='До фильтрации',
                 color='red', edgecolor='black')
        ax2.hist(filtered_data, bins=8, alpha=0.7, label='После фильтрации',
                 color='blue', edgecolor='black')
        ax2.axvline(x=initial_mean, color='red', linestyle='--',
                    label=f'Среднее (до): {initial_mean:.2f}')
        ax2.axvline(x=refined_mean, color='blue', linestyle='--',
                    label=f'Среднее (после): {refined_mean:.2f}')
        ax2.set_xlabel('Значение')
        ax2.set_ylabel('Частота')
        ax2.set_title('Распределение данных')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # График 3: Доверительные интервалы
        ax3 = axes[1, 0]
        intervals = [
            ('95% ПИ\n(для СВ)', lb_95, ub_95, 'green'),
            (f'{self.beta*100:.0f}% ДИ\n(для МО)', lb_beta, ub_beta, 'blue')
        ]
        y_pos = np.arange(len(intervals))

        for i, (label, lb, ub, color) in enumerate(intervals):
            ax3.barh(i, ub - lb, left=lb, height=0.4,
                    color=color, alpha=0.6, edgecolor='black')
            ax3.plot([lb, ub], [i, i], 'o-', color=color, markersize=8, linewidth=2)
            center = (lb + ub) / 2
            ax3.text(center, i, f'[{lb:.2f}, {ub:.2f}]',
                    ha='center', va='bottom', fontsize=9)

        ax3.axvline(x=refined_mean, color='red', linestyle='--',
                   linewidth=2, label=f'x̄* = {refined_mean:.2f}')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([label for label, _, _, _ in intervals])
        ax3.set_xlabel('Значение')
        ax3.set_title('Доверительные интервалы')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')

        # График 4: Боксплот до и после отсеивания
        ax4 = axes[1, 1]
        data_to_plot = [self.original_data, filtered_data]
        bp = ax4.boxplot(data_to_plot, tick_labels=['До фильтрации', 'После фильтрации'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('blue')
        bp['boxes'][1].set_alpha(0.7)
        ax4.set_ylabel('Значение')
        ax4.set_title('Боксплот сравнение')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Сохранение графика
        plt.savefig('statistical_analysis_results.png', dpi=300, bbox_inches='tight')

        plt.show()


def main():
    """Главная функция программы"""
    # Исходные данные для варианта 7
    data = [0.5, 1.8, 3.3, 12.5, 6.9, 7.1, 6.2, 5.7, 4.4, 3.4, 1.6, 0.1]
    beta = 0.89  # Доверительная вероятность
    epsilon_beta = 0.28  # Максимальная вероятная погрешность

    # Создание объекта для анализа
    analysis = StatisticalAnalysis(data, beta, epsilon_beta)

    # Вывод результатов
    analysis.print_results()

    # Визуализация
    analysis.visualize_results()


if __name__ == "__main__":
    main()
