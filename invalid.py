import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 1. чтение CSV чанками
def read_csv_in_chunks(path, chunksize=500):
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


# 2.1 извлечение температурных данных
def extract_columns_1(chunks):
    for chunk in chunks:
        yield chunk[['Station.Location', 'Date.Year', 'Data.Temperature.Avg Temp']]


# 2.2 извлечение данных с большим разбросом данных
def extract_columns_2(chunks):
    for chunk in chunks:
        yield chunk[['Station.State', 'Date.Year', 'Date.Month', 'Data.Temperature.Avg Temp']]


# 2.3 извлекаем колонки по ветру
def extract_columns_3(chunks):
    for chunk in chunks:
        yield chunk[['Station.State', 'Date.Year', 'Date.Month', 'Data.Wind.Speed']]


# 3.1 агрегация: накапливаем сумму температур и количество этих температур по локации для задания 1
def aggregate_temperature(chunks, year_filter=2016):
    stats = {}
    for df in chunks:

        df = df[df["Date.Year"] == year_filter]

        grouped = df.groupby('Station.Location')['Data.Temperature.Avg Temp'].agg(['sum', 'count'])
        for loc, row in grouped.iterrows():
            s, c = float(row['sum']), int(row['count'])
            if loc not in stats:
                stats[loc] = [0.0, 0]
            stats[loc][0] += s
            stats[loc][1] += c

        yield stats


# 3.2 агрегация: копим сумму и количество по ключу (state, month) только за 2016
def aggregate_monthly_temp(chunks, year_filter=2016):
    stats = {}

    for df in chunks:

        df = df[df["Date.Year"] == year_filter]

        grouped = df.groupby(['Station.State', 'Date.Month'])['Data.Temperature.Avg Temp'].agg(['sum', 'count'])

        for (state, month), row in grouped.iterrows():
            s, c = float(row['sum']), int(row['count'])
            if (state, month) not in stats:
                stats[(state, month)] = [0.0, 0]
            stats[(state, month)][0] += s
            stats[(state, month)][1] += c

        yield stats


# 3.3 агрегация: копим сумму и количество значений ветра по штату
#переделать под dataFrame
def aggregate_wind_speed(chunks):
    stats = {}
    for df in chunks:
        grouped = df.groupby('Station.State')['Data.Wind.Speed'].agg(['sum', 'count'])
        for state, row in grouped.iterrows():
            s, c = float(row['sum']), int(row['count'])
            if state not in stats:
                stats[state] = [0.0, 0]
            stats[state][0] += s
            stats[state][1] += c
        yield stats

    # 3.4 агрегация: копим помесячные значения для будущего графика


def aggregate_wind_monthly(chunks):
    stats = {}
    for df in chunks:
        grouped = df.groupby(['Station.State', 'Date.Year', 'Date.Month'])['Data.Wind.Speed'].agg(['sum', 'count'])
        for (state, year, month), row in grouped.iterrows():
            s, c = float(row['sum']), int(row['count'])
            key = (state, int(year), int(month))
            if key not in stats:
                stats[key] = [0.0, 0]
            stats[key][0] += s
            stats[key][1] += c
        yield stats


# 4.1 считаем итоговые средние для задания 1
def finalize_averages(stats_stream):
    final_stats = {}
    for stats in stats_stream:
        final_stats = stats
    result = {}
    for loc, stats in final_stats.items():
        total, count = stats
        if count > 0:
            result[loc] = total / count
    return result


# 4.2 превращаем накопленные суммы в средние значения для задания 2
def compute_monthly_means(stats_stream):
    final_stats = {}
    for stats in stats_stream:
        final_stats = stats

    result = {}
    for key, (total, count) in final_stats.items():
        if count > 0:
            result[key] = total / count
    return result


# 4.3 превращаем накопленные суммы в средние значения для задания 3
def compute_state_wind_means(stats_stream):
    final_stats = {}
    for stats in stats_stream:
        final_stats = stats

    result = {}
    for state, (total, count) in final_stats.items():
        if count > 0:
            result[state] = total / count

    return result


# 5.2 подсчёт стандартного отклонения по штатам
def compute_state_variances(monthly_means):
    state_to_values = {}

    for (state, month), temp in monthly_means.items():
        if state not in state_to_values:
            state_to_values[state] = []
        state_to_values[state].append(temp)

    result = {}
    for state, values in state_to_values.items():
        if len(values) == 12:
            mean_val = float(
                np.mean(values))
            std_val = float(np.std(values))
            result[state] = {"mean": mean_val, "std": std_val, "n": 12}

    return result


# 6.1 график для задания 1
def plot_top_and_bottom(data, top=3, bottom=3):
    sorted_items = sorted(data.items(), key=lambda x: x[1])  # сортируем по температуре
    lowest = sorted_items[:bottom]
    highest = sorted_items[-top:]

    labels = [x[0] for x in lowest + highest]
    values = [x[1] for x in lowest + highest]

    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top-3 и Bottom-3 локаций по среднегодовой температуре")
    plt.ylabel("Средняя температура (°F)")
    plt.tight_layout()
    plt.show(block=False)



# 6.2 график для задания 2
def plot_top_and_bottom_with_ci(state_stats, top=3, bottom=3):
    # сортируем по std (разбросу) элемент - (имя_штата, словарь_со_статистикой)
    sorted_states = sorted(state_stats.items(), key=lambda x: x[1]["std"])
    lowest = sorted_states[:bottom]
    highest = sorted_states[-top:]

    selected = lowest + highest

    labels = [item[0] for item in selected]  # название штата
    means = np.array([item[1]["mean"] for item in selected])  # массив средних знаечний темп
    stds = np.array([item[1]["std"] for item in selected])  # массив ско
    n = 12  # месяцев в году

    ci95 = 1.96 * (stds / np.sqrt(n))  # формула вычесления доверительного инетрвала

    plt.figure(figsize=(11, 5))
    plt.bar(labels, means, yerr=ci95, capsize=6, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title("Top-3 и Bottom-3 штатов по разбросу среднемесячной температуры (2016, с 95% CI)")
    plt.ylabel("Средняя температура (°F)")
    plt.tight_layout()
    plt.show(block=False)



# 6.3 график для задания 3
def plot_windiest_state(monthly_stats, windiest_state, ma_window=4):
    # собираем средние только для нужного штата
    records = []
    for (state, year, month), (s, c) in monthly_stats.items():
        if state == windiest_state and c > 0:
            records.append((int(year), int(month), s / c))

    records.sort(key=lambda x: (x[0], x[1]))  # сортировка по году, потом по месяц
    dates = pd.to_datetime({'year': [r[0] for r in records],
                            'month': [r[1] for r in records],
                            'day': 1})  # даты
    values = np.array([r[2] for r in records])  # средние значение ветра

    # скользящее среднее
    rolling = pd.Series(values).rolling(ma_window, min_periods=1).mean().values

    plt.figure(figsize=(11, 5))
    plt.plot(dates, values, label='Средняя скорость ветра')
    plt.plot(dates, rolling, label=f'Скользящее среднее ({ma_window} мес.)')

    plt.title(f'Скорость ветра в штате {windiest_state}')
    plt.xlabel('Дата')
    plt.ylabel('Скорость ветра (mph)')
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)



def run_pipeline_1(csv_path):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_1(p1)
    p3 = aggregate_temperature(p2)
    result = finalize_averages(p3)
    plot_top_and_bottom(result)


def run_pipeline_2(csv_path):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_2(p1)
    p3 = aggregate_monthly_temp(p2, year_filter=2016)
    p4 = compute_monthly_means(p3)
    state_stats = compute_state_variances(p4)
    plot_top_and_bottom_with_ci(state_stats)


def run_pipeline_3(csv_path, ma_window=4):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_3(p1)
    p3 = aggregate_wind_speed(p2)
    avg_wind = compute_state_wind_means(p3)


    windiest_state = max(avg_wind, key=avg_wind.get)  # находим тот ключ словаря, у которого максимальное значение

    # строим график по месяцам
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_3(p1)
    p4 = aggregate_wind_monthly(p2)

    # берём итоговую версию словаря
    final_monthly = {}
    for stats in p4:
        final_monthly = stats

    plot_windiest_state(final_monthly, windiest_state, ma_window=ma_window)


if __name__ == "__main__":
    start_time = time.time()

    run_pipeline_1('weather.csv')
    run_pipeline_2('weather.csv')
    run_pipeline_3('weather.csv', ma_window=4)

    end_time = time.time()
    print(f"Время выполнения всех пайплайнов: {end_time - start_time:.2f} секунд")