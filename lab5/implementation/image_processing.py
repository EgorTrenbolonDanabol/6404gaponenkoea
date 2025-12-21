"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import cv2

from lab5 import interfaces

import numpy as np





class ImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    """

    def _pad_image(self, image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
        kh, kw = kernel_shape
        pad_h, pad_w = kh // 2, kw // 2

        if len(image.shape) == 3:
            return np.pad(image,
                          ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                          mode='constant')
        else:
            return np.pad(image,
                          ((pad_h, pad_h), (pad_w, pad_w)),
                          mode='constant')

    def _convolve_channel(self, padded_image, kernel: np.ndarray, original_shape: tuple) -> np.ndarray:
        h, w = original_shape
        kh, kw = kernel.shape
        new_image = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                region = padded_image[i:i + kh, j:j + kw]
                new_image[i, j] = np.sum(region * kernel)

        return new_image

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        padded = self._pad_image(image, kernel.shape)
        if len(image.shape) == 3:
            h, w, c = image.shape
            result = np.zeros((h, w, c))
            for channel in range(c):
                result[..., channel] = self._convolve_channel(padded[..., channel], kernel, (h, w))
        else:
            h, w = image.shape
            result = self._convolve_channel(padded, kernel, (h, w))

        return result

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(float)
        red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        gray = np.round(gray)
        return gray.astype(np.uint8)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).
        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        image = 255 * (image / 255) ** (gamma)
        return image.astype(np.uint8)

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Кэнни (cv2.Canny) для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        gray = self._rgb_to_grayscale(image)
        """
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = self._convolution(gray, sobel_x)
        grad_y = self._convolution(gray, sobel_y)
        """

        grad_x, grad_y = self.gradients(gray)

        result_grad = np.sqrt(grad_x.astype(np.float32) ** 2 + grad_y.astype(np.float32) ** 2)

        edges = (result_grad / result_grad.max() * 255).astype(np.uint8)
        binary_edges = (edges > 50).astype(np.uint8) * 255

        return binary_edges

    def gradients(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        return self._convolution(gray, sobel_x), self._convolution(gray, sobel_y)

    def haris_matrix(self, Ix: np.ndarray, Iy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        return self._convolution(Ix * Ix, gaussian_kernel), self._convolution(Iy * Iy, gaussian_kernel), self._convolution(Ix * Iy, gaussian_kernel)

    def formula(self, Ix2: np.ndarray, Iy2: np.ndarray, Ixy: np.ndarray, k: float = 0.04) -> np.ndarray:
        det_M = Ix2 * Iy2 - Ixy * Ixy
        trace_M = Ix2 + Iy2
        return det_M - k * (trace_M ** 2)

    def draw_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        result = image.copy()
        y_coords, x_coords = np.where(corners)
        for y, x in zip(y_coords, x_coords):
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)  # красные точки BGR
        return result

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса (cv2.cornerHarris) для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        gray = self._rgb_to_grayscale(image).astype(np.float32)
        gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        gray = self._convolution(gray, gaussian_kernel)
        Ix, Iy = self.gradients(gray)
        Ix2, Iy2, Ixy = self.haris_matrix(Ix, Iy)
        R = self.formula(Ix2, Iy2, Ixy)
        threshold = 0.01 * R.max()
        corners = R > threshold
        return self.draw_corners(image, corners)

    def cv_edge_detection(self: "ImageProcessing", image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Кэнни (cv2.Canny) для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        gray = self._rgb_to_grayscale(image)
        edges = cv2.Canny(gray, 100, 200)
        return edges
