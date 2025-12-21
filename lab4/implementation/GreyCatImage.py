import cv2
import numpy as np

from lab4.implementation.CatImage import CatImage


class GreyCatImage(CatImage):

    def __init__(self, image: np.ndarray, breed: str, url: str):
        if image.ndim != 2:
            raise ValueError(
                f"Expected a grayscale image (2D), but got a {image.ndim}D image with shape {image.shape}"
            )
        super().__init__(image, breed, url)

    def to_color(self) -> np.ndarray:
        return cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)

    def to_grayscale(self) -> np.ndarray:
        return  self._image
