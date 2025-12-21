import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from parquetCheck import benchmark_csv_vs_parquet

PARQUET_PATH = "weather.parquet"



def csv_to_parquet_if_needed(csv_path):
    if os.path.exists(PARQUET_PATH):
        print("Parquet уже существует — пропускаю.")
        return

    print("Создаю Parquet из CSV...")
    df = pd.read_csv(csv_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, PARQUET_PATH)
    print("Parquet создан.")


def plot_wind_vs_precipitation():
    # Чтение только нужных столбцов
    df = pq.read_table(PARQUET_PATH,
                       columns=['Station.Location', 'Data.Wind.Speed', 'Data.Precipitation']).to_pandas()

    # Удаляем пропуски
    df = df.dropna(subset=['Data.Wind.Speed', 'Data.Precipitation'])

    # Группируем по локациям и считаем средние значения
    agg = df.groupby('Station.Location').mean().reset_index()

    # Рассчитываем корреляцию
    corr = agg['Data.Wind.Speed'].corr(agg['Data.Precipitation'])
    print(f"Коэффициент корреляции Пирсона между ветром и осадками: {corr:.3f}")

    # Строим scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(agg['Data.Wind.Speed'], agg['Data.Precipitation'])
    plt.xlabel('Средняя скорость ветра (Wind.Speed)')
    plt.ylabel('Среднее количество осадков (Precipitation)')
    plt.title(f'Корреляция между ветром и осадками: r = {corr:.3f}')
    plt.grid(True)
    plt.show()

def aggregate_temperature_df(df, year_filter=2016):
    df = df[df['Date.Year'] == year_filter]
    g = df.groupby('Station.Location')['Data.Temperature.Avg Temp']
    res = g.agg(['sum', 'count']).reset_index()
    res['avg_temp'] = res['sum'] / res['count']
    return res


def compute_state_variances_df(df):
    stats = df.groupby('Station.State')['avg_temp'].agg(['mean', 'std', 'count']).reset_index()
    return stats[stats['count'] == 12]


def aggregate_monthly_temp_df(df, year_filter=2016):
    df = df[df['Date.Year'] == year_filter]
    g = df.groupby(['Station.State', 'Date.Month'])['Data.Temperature.Avg Temp']
    res = g.agg(['sum', 'count']).reset_index()
    res['avg_temp'] = res['sum'] / res['count']
    return res


def aggregate_wind_speed_df(df):
    g = df.groupby('Station.State')['Data.Wind.Speed']
    res = g.agg(['sum', 'count']).reset_index()
    res['avg_wind'] = res['sum'] / res['count']
    return res


def aggregate_wind_monthly_df(df):
    g = df.groupby(['Station.State', 'Date.Year', 'Date.Month'])['Data.Wind.Speed']
    res = g.agg(['sum', 'count']).reset_index()
    res['avg_wind'] = res['sum'] / res['count']
    return res



def plot_top_and_bottom_df(df, top=3, bottom=3):
    df_sorted = df.sort_values('avg_temp')
    selected = pd.concat([df_sorted.head(bottom), df_sorted.tail(top)])

    plt.figure(figsize=(11,5))
    plt.bar(selected['Station.Location'], selected['avg_temp'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=False)


def plot_top_and_bottom_with_ci_df(df, top=3, bottom=3):
    sorted_df = df.sort_values('std')
    selected = pd.concat([sorted_df.head(bottom), sorted_df.tail(top)])
    ci95 = 1.96 * (selected['std'] / np.sqrt(selected['count']))

    plt.figure(figsize=(11,5))
    plt.bar(selected['Station.State'], selected['mean'], yerr=ci95, capsize=6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=False)


def plot_windiest_state_df(df, state, ma_window=4):
    st = df[df['Station.State'] == state].sort_values(['Date.Year', 'Date.Month'])
    dates = pd.to_datetime(dict(year=st['Date.Year'], month=st['Date.Month'], day=1))

    values = st['avg_wind'].values
    rolling = pd.Series(values).rolling(ma_window, min_periods=1).mean()

    plt.figure(figsize=(11,5))
    plt.plot(dates, values)
    plt.plot(dates, rolling)
    plt.tight_layout()
    plt.show(block=False)



def run_pipeline_1():
    cols = ['Station.Location', 'Date.Year', 'Data.Temperature.Avg Temp']
    df = pd.read_parquet(PARQUET_PATH, columns=cols)

    agg = aggregate_temperature_df(df)
    plot_top_and_bottom_df(agg)



def run_pipeline_2():
    cols = ['Station.State', 'Date.Year', 'Date.Month', 'Data.Temperature.Avg Temp']
    df = pd.read_parquet(PARQUET_PATH, columns=cols)

    monthly = aggregate_monthly_temp_df(df)
    stats = compute_state_variances_df(monthly)
    plot_top_and_bottom_with_ci_df(stats)



def run_pipeline_3(ma_window=4):
    cols = ['Station.State', 'Data.Wind.Speed']
    df = pd.read_parquet(PARQUET_PATH, columns=cols)
    agg = aggregate_wind_speed_df(df)

    windiest = agg.loc[agg['avg_wind'].idxmax(), 'Station.State']

    cols2 = ['Station.State', 'Date.Year', 'Date.Month', 'Data.Wind.Speed']
    df2 = pd.read_parquet(PARQUET_PATH, columns=cols2)

    monthly = aggregate_wind_monthly_df(df2)

    plot_windiest_state_df(monthly, windiest, ma_window)


if __name__ == '__main__':
    csv_path = "weather.csv"

    csv_to_parquet_if_needed(csv_path)

    benchmark_csv_vs_parquet(csv_path)

    plot_wind_vs_precipitation()

    start_time = time.time()
    run_pipeline_1()
    run_pipeline_2()
    run_pipeline_3(ma_window=4)
    end_time = time.time()
    print(f"Время выполнения всех пайплайнов: {end_time - start_time:.2f} секунд")
