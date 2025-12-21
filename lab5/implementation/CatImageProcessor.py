import logging
import os
import time
import requests
import numpy as np
import cv2
from dotenv import load_dotenv
from typing import List
from lab5.implementation.CatImage import CatImage
from lab5.implementation.ColorCatImage import ColorCatImage
from lab5.implementation.GreyCatImage import GreyCatImage

def timeit(method):
    """Декоратор для замера времени выполнения."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{method.__name__} executed in {elapsed:.3f} sec.")
        return result
    return wrapper

class CatImageProcessor:
    BASE_URL = "https://api.thecatapi.com/v1/images/search"

    def __init__(self, save_dir: str = "results"):
        load_dotenv()
        self._api_key = os.getenv("API_KEY")
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @timeit
    def fetch_images(self, limit) -> List[CatImage]:
        """Скачивает изображения с TheCatAPI."""
        headers = {"x-api-key": self._api_key}
        params = {"limit": limit, "has_breeds": 1}
        resp = requests.get(self.BASE_URL, headers=headers, params=params)

        if 200 <= resp.status_code < 300:
            data = resp.json()
        else:
            raise RuntimeError(f"return code from server {resp.status_code}")

        cat_images = []

        for item in data:

            url = item["url"]
            breed = item["breeds"][0]["name"]


            img_resp = requests.get(url)
            img_arr = np.asarray(bytearray(img_resp.content), dtype=np.uint8)

            img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)

            if img.ndim == 3:
                cat = ColorCatImage(img, breed, url)
            else:
                cat = GreyCatImage(img, breed, url)

            cat_images.append(cat)
            logging.info(f"fetched image {breed} ({url})")

        return cat_images



    def save_image(self, method_name, idx, breed_name, func):
        result = func()
        path = os.path.join(self._save_dir, f"{idx}_{breed_name}_{method_name}.png")
        try:
            cv2.imwrite(path, result.image)
            logging.info(f"{method_name} saved: {path}")
        except Exception as e:
            logging.ERROR(f"error while saving {method_name}: {e}")
        return result

    @timeit
    def process_and_save(self, cats: List[CatImage]):
        idx = 1  # стартовый индекс
        for cat in cats:
            breed_name = cat.breed.replace(" ", "_")
            # === 1. ОРИГИНАЛ ===
            self.save_image("original", idx, breed_name, lambda c=cat: c)
            # === 2–5. Методы обработки ===
            self.save_image("cv_edges", idx, breed_name, cat.edge_detection_cv)
            self.save_image("custom_edges", idx, breed_name, cat.edge_detection_custom)
            self.save_image("cv_corners", idx, breed_name, cat.corner_detection_cv)
            self.save_image("custom_corners", idx, breed_name, cat.corner_detection_custom)
            logging.info(f"All images was saved for {cat.breed}")
            idx+=1
