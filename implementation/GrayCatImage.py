import cv2
import numpy as np
from implementation import CatImage, ImageProcessing


class GrayCatImage(CatImage):

    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):

        if len(image_data.shape) == 3:
            image_data = ImageProcessing()._rgb_to_grayscale(image_data)

        gray_image = image_data
        super().__init__(gray_image, image_url, breed)

    def to_color(self) -> np.ndarray:
        return cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)