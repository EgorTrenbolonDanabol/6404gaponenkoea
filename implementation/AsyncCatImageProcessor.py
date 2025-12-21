import asyncio
import aiofiles
import os
import time
import cv2
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
from typing import List, Tuple
from dotenv import load_dotenv
import aiohttp
import numpy as np
from implementation import ColorCatImage, GrayCatImage


def process_one(args):
    idx, breed, url, img = args
    print(f"[LOG] Convolution start {idx}: {breed}:PID {os.getpid()}")

    if img.ndim == 3:
        cat = ColorCatImage(img, breed, url)
    else:
        cat = GrayCatImage(img, breed, url)

    edges_cv = cat.cv_edge_detection().image
    edges_custom = cat.my_edge_detection().image
    print(f"[LOG] Convolution end  {idx}: {breed}:PID {os.getpid()}")

    return idx, breed, img, edges_cv, edges_custom


def async_timeit(method):
    """Асинхронный декоратор для замера времени выполнения."""

    @wraps(method)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await method(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[LOG] {method.__name__} выполнен за {elapsed:.3f} сек.")
        return result

    return wrapper


class AsyncCatImageProcessor:
    """Работает с API, управляет обработкой и сохранением изображений."""

    BASE_URL = "https://api.thecatapi.com/v1/images/search"

    def __init__(self, save_dir: str = "results"):
        load_dotenv()
        self._api_key = os.getenv("API_KEY")
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @async_timeit
    async def fetch_images(self, limit: int = 3) -> List[Tuple[int, str, str, np.ndarray]]:
        """Полуачем ответ от сервера в виде json"""
        headers = {"x-api-key": self._api_key}
        params = {"limit": limit, "has_breeds": 1}
        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, headers=headers, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            # фиксируем порядок индексов сразу
            index_items = []
            for i, item in enumerate(data, start=1):
                url = item.get("url")
                breed = (item.get("breeds"))[0].get("name")
                index_items.append((i, url, breed))

            # создаём корутины на скачивание
            tasks = [self._download_and_decode(session, idx, url, breed) for (idx, url, breed) in index_items]

            # запускаем конкурентно и ждём всех
            results = await asyncio.gather(*tasks)
            return results

    async def _download_and_decode(self, session: aiohttp.ClientSession, idx: int, url: str, breed: str):
        """Асинхронно скачивает одну картинку, декодирует и возвращает CatImage."""

        print(f"[LOG] Downloading image {idx} started: {breed} ({url})")

        # делаем асинхронный HTTP-запрос
        async with session.get(url) as resp:
            resp.raise_for_status()
            content = await resp.read()

        img_arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)

        print(f"[LOG] Downloading image {idx} finished: {breed}")
        return idx, breed, url, img

    @async_timeit
    async def process_and_save(self, cats: List[tuple]):
        save_dir = self._save_dir

        loop = asyncio.get_running_loop()
        processed = []
        # 1.параллельная CPU-обработка
        with ProcessPoolExecutor() as pool:
            # Передаём функцию + данные
            futures = [loop.run_in_executor(pool, process_one, item) for item in cats]
            for i in asyncio.as_completed(futures):
                processed.append(await i)

        # 2. асинхронное сохранение
        for idx, breed, orig_img, edges_cv, edges_custom in processed:
            breed_name = (breed).replace(" ", "_").lower()

            # оригинал
            orig_path = os.path.join(save_dir, f"{idx}_{breed_name}_original.png")
            ok, buf = cv2.imencode(".png", orig_img)
            if ok:
                async with aiofiles.open(orig_path, "wb") as f:
                    await f.write(buf.tobytes())

            # библиотечная функция
            cv_path = os.path.join(save_dir, f"{idx}_{breed_name}_cv.png")
            ok, buf = cv2.imencode(".png", edges_cv)
            if ok:
                async with aiofiles.open(cv_path, "wb") as f:
                    await f.write(buf.tobytes())

            # ручная реализация
            custom_path = os.path.join(save_dir, f"{idx}_{breed_name}_custom.png")
            ok, buf = cv2.imencode(".png", edges_custom)
            if ok:
                async with aiofiles.open(custom_path, "wb") as f:
                    await f.write(buf.tobytes())

            print(f"[LOG] saved all versions for {breed} (#{idx})")