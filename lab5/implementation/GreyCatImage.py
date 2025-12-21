import cv2
import numpy as np

from lab5.implementation import ImageProcessing
from lab5.implementation.CatImage import CatImage


class GreyCatImage(CatImage):
    """Класс для работы с черно-белыми изображениями кошек."""

    def __init__(self, image: np.ndarray, breed: str, url: str):
        # Если цветное изображение, конвертируем в grayscale
        if image.ndim == 3:
            image = ImageProcessing()._rgb_to_grayscale(image)
        elif image.ndim != 2:
            raise ValueError(
                f"Expected 2D grayscale image, got {image.ndim}D with shape {image.shape}"
            )
        super().__init__(image, breed, url)

    def to_color(self) -> np.ndarray:
        """Преобразует ч/б изображение в цветное (BGR)."""
        return cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)

    def to_grayscale(self) -> np.ndarray:
        """Возвращает исходное ч/б изображение."""
        return self._image
