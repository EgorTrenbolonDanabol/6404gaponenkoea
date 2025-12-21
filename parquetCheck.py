import os
import time

import pandas as pd

PARQUET_PATH = "weather.parquet"

def benchmark_csv_vs_parquet(csv_path):
    print("\nСравниваю скорость чтения:")

    # ---- 1. CSV полностью ----
    t1 = time.time()
    pd.read_csv(csv_path)
    t2 = time.time()
    csv_time = t2 - t1

    # ---- 2. Parquet полностью ----
    t1 = time.time()
    pd.read_parquet(PARQUET_PATH)
    t2 = time.time()
    pq_time = t2 - t1

    # ---- 3. CSV только нужные столбцы ----
    needed_cols = [
        'Data.Temperature.Avg Temp',
        'Data.Wind.Speed',
        'Date.Year',
        'Station.State'
    ]

    t1 = time.time()
    pd.read_csv(csv_path, usecols=needed_cols)
    t2 = time.time()
    csv_cols_time = t2 - t1

    # ---- 4. Parquet только нужные столбцы ----
    t1 = time.time()
    pd.read_parquet(PARQUET_PATH, columns=needed_cols)
    t2 = time.time()
    pq_cols_time = t2 - t1

    # ---- вывод ----
    print(f"CSV (всё):                     {csv_time:.3f} сек")
    print(f"Parquet (всё):                 {pq_time:.3f} сек")
    print(f"CSV (только столбцы):          {csv_cols_time:.3f} сек")
    print(f"Parquet (только столбцы):      {pq_cols_time:.3f} сек")

    print("\nУскорение Parquet:")
    print(f"Полное чтение:     ×{csv_time/pq_time:.1f} раз")
    print(f"По столбцам:        ×{csv_cols_time/pq_cols_time:.1f} раз\n")
