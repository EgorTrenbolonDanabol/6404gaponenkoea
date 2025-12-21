from abc import ABC
import numpy as np
import cv2
from implementation import ImageProcessing


class CatImage(ABC):
    def __init__(self, image: np.ndarray, breed: str, url: str):
        self._image = image
        self._breed = breed
        self._url = url
        self._processor = ImageProcessing()

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def breed(self) -> str:
        return self._breed

    @property
    def url(self) -> str:
        return self._url

    def __str__(self) -> str:
        return f"CatImage(breed={self._breed}, url={self._url})"

    def __add__(self, other: "CatImage") -> np.ndarray:

        img_self = self._image
        img_other = other._image

        if len(img_self.shape) == 2 and len(img_other.shape) == 3:
            img_other = other.to_grayscale()

        if len(img_self.shape) == 3 and len(img_other.shape) == 2:
            img_other = other.to_color()

        if img_self.shape != img_other.shape:
            height, width = img_self.shape[:2]
            img_other = cv2.resize(img_other, (width, height))

        result = img_self.astype(np.int16) + img_other.astype(np.int16)
        return np.clip(result, 0, 255).astype(np.uint8)

    def __sub__(self, other: "CatImage") -> np.ndarray:

        img_self = self._image
        img_other = other._image
        print(f"[LOG] До начала преобразований")
        print(f"[LOG] max Уменьшаемое {img_self.max()}")
        print(f"[LOG] max Вычитаемое {img_other.max()}")

        if img_self.ndim == 2 and img_other.ndim == 3:
            img_other = other.to_grayscale()

        if img_self.ndim == 3 and img_other.ndim == 2:
            img_other = other.to_color()

        if img_self.shape != img_other.shape:
            height, width = img_self.shape[:2]
            img_other = cv2.resize(img_other, (width, height))

        result = img_self.astype(np.float32) - img_other.astype(np.float32)

        result_min = result.min()
        result_sub = result - result_min
        result_max = result.max()
        result_division = result_sub / result_max
        result_mul = result_division * 255
        return result_mul.astype(np.uint8)

    def cv_edge_detection(self) -> "CatImage":
        from implementation.GrayCatImage import GrayCatImage
        edge = self._processor.cv_edge_detection(self._image)
        return  GrayCatImage(edge, self._breed, self._url)

    def my_edge_detection(self) -> "CatImage":
        from implementation.GrayCatImage import GrayCatImage
        edge = self._processor.edge_detection(self._image)
        return  GrayCatImage(edge, self._breed, self._url)
