import os
import time
import requests
import numpy as np
import cv2
from dotenv import load_dotenv
from typing import List

from lab5.implementation import CatImage, GrayCatImage, ColorCatImage


def timeit(method):
    """Декоратор для замера времени выполнения."""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[LOG] {method.__name__} выполнен за {elapsed:.3f} сек.")
        return result

    return wrapper


class CatImageProcessor:
    """Работает с API, управляет обработкой и сохранением изображений."""

    BASE_URL = "https://api.thecatapi.com/v1/images/search"

    def __init__(self, save_dir: str = "results"):
        load_dotenv()
        self._api_key = os.getenv("API_KEY")
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @timeit
    def fetch_images(self, limit: int = 3) -> List[CatImage]:
        """Скачивает изображения с TheCatAPI."""
        headers = {"x-api-key": self._api_key}
        params = {"limit": limit, "has_breeds": 1}
        resp = requests.get(self.BASE_URL, headers=headers, params=params)
        resp.raise_for_status()

        data = resp.json()
        cat_images = []

        for item in data:
            url = item["url"]
            breed = item["breeds"][0]["name"] if item.get("breeds") else "unknown"

            img_resp = requests.get(url)
            img_arr = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
            if img.ndim == 3:
                cat = ColorCatImage(img, breed, url)
                img_type = "Color"
            else:
                cat = GrayCatImage(img, breed, url)
                img_type = "Grayscale"

            cat_images.append(cat)
            print(f"[LOG] Скачано изображение: {breed}, тип: {img_type}, url: {url}")

        return cat_images

    @timeit
    def process_and_save(self, cats: List[CatImage]):
        """Обрабатывает и сохраняет изображения."""
        for idx, cat in enumerate(cats, start=1):
            breed_name = (cat.breed or "unknown").replace(" ", "_").lower()

            # оригинал
            orig_path = os.path.join(self._save_dir, f"{idx}_{breed_name}_original.png")
            ok = cv2.imwrite(orig_path, cat.image)

            # библиотечный метод
            edges_cv = cat.cv_edge_detection()
            cv_path = os.path.join(self._save_dir, f"{idx}_{breed_name}_cv.png")
            ok = cv2.imwrite(cv_path, edges_cv.image)

            # ручная реализация
            edges_custom = cat.my_edge_detection()
            custom_path = os.path.join(self._save_dir, f"{idx}_{breed_name}_my.png")
            ok = cv2.imwrite(custom_path, edges_custom.image)

