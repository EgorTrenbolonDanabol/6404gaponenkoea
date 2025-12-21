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

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Использует функцию cv2.filter2D для применения ядра к изображению.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """

        kernel_height, kernel_width = kernel.shape

        padding_height = kernel_height // 2
        padding_width = kernel_width // 2

        if len(image.shape) == 3:
            # (0,0) означает не добавлять ничего по каналам
            padded_image = np.pad(image, (
                (padding_height, padding_height),
                (padding_width, padding_width),
                (0, 0)),
                                  mode='constant')

            height, width, canals = image.shape
            new_image = np.zeros((height, width, canals), dtype=np.float32)
            for i in range(height):
                for j in range(width):
                    for x in range(canals):
                        new_image[i][j] = np.sum(
                            kernel * padded_image[i:i + kernel_height, j:j + kernel_width, x]

                        )
        else:
            padded_image = np.pad(image,
                                  ((padding_height, padding_height),
                                   (padding_width, padding_width)),
                                  mode='constant')

            height, width = image.shape
            new_image = np.zeros((height, width), dtype=np.float32)
            for i in range(height):
                for j in range(width):
                    new_image [i][j]= np.sum(
                        kernel * padded_image[i:i + kernel_height, j:j +kernel_width]

                    )


        return new_image
# формула ярксти, в итоге получаются сервые оттенки
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 4:
            image = image[:, :, :3]  # убираем альфа-канал (прозрачность)
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

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
        # чтоб работать только с одним каналом, черно белым
        gray = self._rgb_to_grayscale(image)


        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = self._convolution(gray, sobel_x)
        grad_y = self._convolution(gray, sobel_y)

        result_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

        #чтобы только черно белое было
        edges = (result_grad / result_grad.max() * 255).astype(np.uint8)
        binary_edges = (edges > 50).astype(np.uint8) * 255

        return binary_edges

    def library_edge_detection(self, image: np.ndarray) -> np.ndarray:
        gray = self._rgb_to_grayscale(image)
        edges = cv2.Canny(gray, 100, 200)
        return edges

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
        gray = self._rgb_to_grayscale(image)

        # Вычисляем градиенты
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        Ix = self._convolution(gray, sobel_x)
        Iy = self._convolution(gray, sobel_y)


        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy

        # Гауссово размытие для убирания шума
        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16.0

        Ix2_smooth = self._convolution(Ix2, gaussian_kernel)
        Iy2_smooth = self._convolution(Iy2, gaussian_kernel)
        Ixy_smooth = self._convolution(Ixy, gaussian_kernel)

        #ЧДУ
        k = 0.04
    #сильные изменениях в обоих направлениях
        det_M = Ix2_smooth * Iy2_smooth - Ixy_smooth * Ixy_smooth
        trace_M = Ix2_smooth + Iy2_smooth
        R = det_M - k * (trace_M ** 2)

        threshold = 0.01 * R.max()
        result_image = image.copy()

        y_coords, x_coords = np.where(R > threshold)

        for i in range(len(y_coords)):
            y = y_coords[i]
            x = x_coords[i]
            cv2.circle(result_image, (x, y), 3, (0, 0, 0), -1)

        return result_image



    def library_corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса (cv2.cornerHarris) для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        gray = self._rgb_to_grayscale(image)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        result = image.copy()
        result[dst > 0.01 * dst.max()] = [255, 0, 0]
        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")
