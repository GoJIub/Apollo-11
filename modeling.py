import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Константы

G = 6.67430e-11  # гравитационная постоянная, м^3⋅кг^−1⋅с^−2
M_moon = 7.347673e22  # масса Луны, кг
R_moon = 1738.4e3  # радиус Луны, м

# Загрузка данных из CSV файла
data = pd.read_csv("data.csv")

# Извлекаем интересующие нас данные

time = data['Time'].values  # время
thrust = data['Thrust'].values  # тяга
mass = data['Mass'].values  # масса
real_acceleration = data['Acceleration'].values  # реальное ускорение

# Фильтрация данных по времени (с 1455 до 1540 секунд)

time_filtered_idx = (time >= 1455) & (time <= 1535)
time_filtered = [i - 1455 for i in time[time_filtered_idx]]
real_acceleration_filtered = real_acceleration[time_filtered_idx]
thrust_filtered = thrust[time_filtered_idx]
mass_filtered = mass[time_filtered_idx]

# --- Фильтр Баттерворта ---

def butter_filter(data, cutoff=0.05, fs=1, order=4):
    # Нормализуем частоту среза
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # Создаем фильтр Баттерворта
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Применяем фильтр
    return filtfilt(b, a, data)

# Сглаживаем реальное ускорение
smoothed_real_acceleration = butter_filter(real_acceleration_filtered)

# --- Расчетное ускорение ---

# Сила тяжести на каждом шаге времени
F_gravity = (G * M_moon) / (R_moon**2)  # сила тяжести (для упрощения, принимаем постоянной)
calculated_acceleration = (thrust_filtered - F_gravity) / mass_filtered

# Сглаживаем расчетное ускорение
smoothed_calculated_acceleration = butter_filter(calculated_acceleration)

# --- Разница между сглаженными реальным и расчетным ускорением ---
acceleration_difference = smoothed_real_acceleration - smoothed_calculated_acceleration

# --- Подсчёт погрешностей ---

abs_m, rel_m = 0, 0
abs_avg, rel_avg = 0, 0

for i in time_filtered:

    # Значения ускорений в данный момент времени
    real = smoothed_real_acceleration[i]
    calc = smoothed_calculated_acceleration[i]

    # Значения погрешностей в данный момент времени
    abs_cur = abs(real - calc)
    rel_cur = abs_cur / real * 100

    # Переопределение максимальных погрешностей
    abs_m = max(abs_m, abs_cur)
    rel_m = max(rel_m, rel_cur)

    # Подсчёт средних погрешностей
    abs_avg += abs_cur
    rel_avg += rel_cur

# Вывод результатов

print(f'Max absolute error: {abs_m:.3} м/с^2\nMax relative error: {rel_m:.3}%')
print(f'Average absolute error: {(abs_avg / len(time_filtered)):.3} м/с^2\nAverage relative error: {(rel_avg / len(time_filtered)):.3}%')

# --- Построение графиков ---

# Строим график с тремя линиями
plt.figure(figsize=(12, 8))

# Реальное ускорение
plt.plot(time_filtered, real_acceleration_filtered, label="Реальное ускорение", color='gray', alpha=0.5)

# Сглаженное реальное ускорение
plt.plot(time_filtered, smoothed_real_acceleration, label="Сглаженное реальное ускорение", color='b')

# Сглаженное расчетное ускорение
plt.plot(time_filtered, smoothed_calculated_acceleration, label="Сглаженное расчетное ускорение", color='g')

# Разница
plt.plot(time_filtered, acceleration_difference, label="Разница между сглаженными ускорениями", color='r', linestyle='--')

# Настройка графика

plt.xlabel("Время (с)")
plt.ylabel("Ускорение (м/с²)")
plt.title("Сравнение реального и расчетного ускорения с их разницей")
plt.legend()
plt.grid(True)

# Показываем график
plt.show()
