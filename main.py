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


# 3.1 Агрегация средних температур по локациям (DataFrame)
def aggregate_temperature_df(chunks, year_filter=2016):

    gen = (chunk[chunk['Date.Year'] == year_filter]
             .groupby('Station.Location')['Data.Temperature.Avg Temp']
             .agg(['sum', 'count'])
             .reset_index()
         for chunk in chunks)

    full = pd.concat( gen
        ,
        ignore_index=True
    )
    full = full.groupby('Station.Location').sum().reset_index()
    full['avg_temp'] = full['sum'] / full['count']
    return full


# 3.2 Агрегация среднемесячной температуры (DataFrame)
def aggregate_monthly_temp_df(chunks, year_filter=2016):
    full = pd.concat(
        (chunk[chunk['Date.Year'] == year_filter]
             .groupby(['Station.State', 'Date.Month'])['Data.Temperature.Avg Temp']
             .agg(['sum', 'count'])
             .reset_index()
         for chunk in chunks),
        ignore_index=True
    )

    full = full.groupby(['Station.State', 'Date.Month']).sum().reset_index()
    full['avg_temp'] = full['sum'] / full['count']
    return full


# 3.3 Средняя скорость ветра по штатам (DataFrame)
def aggregate_wind_speed_df(chunks):
    full = pd.concat(
        (chunk.groupby('Station.State')['Data.Wind.Speed']
             .agg(['sum', 'count'])
             .reset_index()
         for chunk in chunks),
        ignore_index=True
    )

    full = full.groupby('Station.State').sum().reset_index()
    full['avg_wind'] = full['sum'] / full['count']
    return full


# 3.4 Агрегация помесячной скорости ветра
def aggregate_wind_monthly_df(chunks):
    full = pd.concat(
        (chunk.groupby(['Station.State', 'Date.Year', 'Date.Month'])['Data.Wind.Speed']
             .agg(['sum', 'count'])
             .reset_index()
         for chunk in chunks),
        ignore_index=True
    )

    full = full.groupby(['Station.State', 'Date.Year', 'Date.Month']).sum().reset_index()
    full['avg_wind'] = full['sum'] / full['count']
    return full


# 5.2 variance
def compute_state_variances_df(df):
    stats = df.groupby('Station.State')['avg_temp'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats[stats['count'] == 12]
    return stats


def plot_top_and_bottom_df(df, top=3, bottom=3):
    df_sorted = df.sort_values('avg_temp')
    selected = pd.concat([df_sorted.head(bottom), df_sorted.tail(top)])

    plt.figure(figsize=(11, 5))
    plt.bar(selected['Station.Location'], selected['avg_temp'])
    plt.xticks(rotation=45, ha='right')
    plt.title("Top-3 и Bottom-3 локаций по среднегодовой температуре")
    plt.ylabel("Средняя температура (°F)")
    plt.tight_layout()
    plt.show(block=False)


def plot_top_and_bottom_with_ci_df(df, top=3, bottom=3):
    sorted_df = df.sort_values('std')
    selected = pd.concat([sorted_df.head(bottom), sorted_df.tail(top)])

    ci95 = 1.96 * (selected['std'] / np.sqrt(selected['count']))

    plt.figure(figsize=(11, 5))
    plt.bar(selected['Station.State'], selected['mean'], yerr=ci95, capsize=6)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top-3 и Bottom-3 штатов по std температур (95% CI)")
    plt.tight_layout()
    plt.show(block=False)


def plot_windiest_state_df(df, state, ma_window=4):
    st = df[df['Station.State'] == state].sort_values(['Date.Year', 'Date.Month'])
    dates = pd.to_datetime(dict(year=st['Date.Year'], month=st['Date.Month'], day=1))

    values = st['avg_wind'].values
    rolling = pd.Series(values).rolling(ma_window, min_periods=1).mean()

    plt.figure(figsize=(11, 5))
    plt.plot(dates, values, label='Средняя скорость ветра')
    plt.plot(dates, rolling, label=f'Скользящее среднее {ma_window}')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- PIPELINES ----------

def run_pipeline_1(csv_path):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_1(p1)
    df = aggregate_temperature_df(p2)
    plot_top_and_bottom_df(df)


def run_pipeline_2(csv_path):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_2(p1)
    df = aggregate_monthly_temp_df(p2)
    stats = compute_state_variances_df(df)
    plot_top_and_bottom_with_ci_df(stats)


def run_pipeline_3(csv_path, ma_window=4):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_3(p1)
    wind_df = aggregate_wind_speed_df(p2)

    windiest = wind_df.loc[wind_df['avg_wind'].idxmax(), 'Station.State']

    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns_3(p1)
    monthly = aggregate_wind_monthly_df(p2)

    plot_windiest_state_df(monthly, windiest, ma_window=ma_window)

if __name__ == '__main__':
    start_time = time.time()

    run_pipeline_1('weather.csv')
    run_pipeline_2('weather.csv')
    run_pipeline_3('weather.csv', ma_window=4)

    end_time = time.time()
    print(f"Время выполнения всех пайплайнов: {end_time - start_time:.2f} секунд")
