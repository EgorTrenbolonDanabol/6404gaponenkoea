import numpy as np
import cv2

from implementation import CatImage


class ColorCatImage(CatImage):

    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):

        if len(image_data.shape) == 2:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)

        gray_image = image_data
        super().__init__(gray_image, image_url, breed)

    def to_grayscale(self) -> np.ndarray:
        """Доп.метод: вернуть чб изображение."""
        return self._processor._rgb_to_grayscale(self._image)