import numpy as np
import cv2

from lab5.implementation import ImageProcessing
from abc import ABC, abstractmethod


class CatImage(ABC):
    """Класс, инкапсулирующий изображение кота и методы обработки."""

    def __init__(self, image: np.ndarray, breed: str, url: str):
        self._image = image  # картинка
        self._breed = breed  # порода
        self._url = url  # ссылка
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

    def edge_detection_cv(self) -> "CatImage":
        from lab5.implementation.GreyCatImage import GreyCatImage
        return GreyCatImage(self._processor.library_edge_detection(self._image), self._breed, self._url)

    def edge_detection_custom(self) -> "CatImage":
        from lab5.implementation.GreyCatImage import GreyCatImage
        return GreyCatImage(self._processor.edge_detection(self._image), self._breed, self._url)

    def corner_detection_custom(self) -> "CatImage":
        from lab5.implementation.ColorCatImage import ColorCatImage
        from lab5.implementation.GreyCatImage import GreyCatImage
        result_img = self._processor.corner_detection(self._image)
        if result_img.ndim == 2:
            return GreyCatImage(result_img, self._breed, self._url)
        else:
            return ColorCatImage(result_img, self._breed, self._url)

    def corner_detection_cv(self) -> "CatImage":
        from lab5.implementation.ColorCatImage import ColorCatImage
        from lab5.implementation.GreyCatImage import GreyCatImage
        result_img = self._processor.library_corner_detection(self._image)
        if result_img.ndim == 2:  # серое изображение
            return GreyCatImage(result_img, self._breed, self._url)
        else:
            return ColorCatImage(result_img, self._breed, self._url)

    def __add__(self, other: "CatImage") -> np.ndarray:
        if not isinstance(other, CatImage):
            return NotImplemented
        if self._image.ndim == 2 and other._image.ndim == 3:
            other._image = other.to_grayscale()

        if self._image.ndim == 3 and other._image.ndim == 2:
            other._image = other.to_color()

        if self._image.shape != other._image.shape:
            h = self._image.shape[0]
            w = self._image.shape[1]
            other._image = cv2.resize(other._image, (w, h))

        result = self._image.astype(np.int16) + other._image.astype(np.int16)

        return np.clip(result, 0, 255).astype(np.uint8)

    def __sub__(self, other: "CatImage") -> np.ndarray:

        if not isinstance(other, CatImage):
            return NotImplemented

        if self._image.ndim == 2 and other._image.ndim == 3:
            other._image = other.to_grayscale()

        if self._image.ndim == 3 and other._image.ndim == 2:
            other._image = other.to_color()

        if self._image.shape != other._image.shape:
            h = self._image.shape[0]
            w = self._image.shape[1]
            other._image = cv2.resize(other._image, (w, h))



        result = self._image.astype(np.int16) - other._image.astype(np.int16)
        return np.clip(result, 0, 255).astype(np.uint8)

    @abstractmethod
    def to_grayscale(self) -> np.ndarray:
        pass

    @abstractmethod
    def to_color(self) -> np.ndarray:
        pass
