import numpy as np

from lab5.implementation.CatImage import CatImage


class ColorCatImage(CatImage):

    def __init__(self, image: np.ndarray, breed: str, url: str):
        if image.ndim != 3:
            raise ValueError(
                f"Ожидалось цветное изображение, получено {image.ndim}D с формой {image.shape}"
            )
        super().__init__(image, breed, url)

    def to_grayscale(self) -> np.ndarray:
        return self._processor._rgb_to_grayscale(self._image)

    def to_color(self) -> np.ndarray:
        return self._image