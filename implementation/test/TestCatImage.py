import unittest
import numpy as np

from implementation.CatImage import CatImage
from implementation.ColorCatImage import ColorCatImage
from implementation.GrayCatImage import GrayCatImage
import numpy as np
import cv2
import pytest

from implementation.image_processing import ImageProcessing
from implementation.CatImage import CatImage
from implementation.ColorCatImage import ColorCatImage
from implementation.GrayCatImage import GrayCatImage


class TestCatImage(unittest.TestCase):

    def sample_color(self):
        """3x3 RGB-картинка."""
        return np.array([
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[15, 25, 35], [45, 55, 65], [75, 85, 95]],
            [[20, 30, 40], [50, 60, 70], [80, 90, 100]],
        ], dtype=np.uint8)

    def sample_gray(self):
        """3x3 grayscale-картинка."""
        return np.array([
            [10, 40, 70],
            [15, 45, 75],
            [20, 50, 80]
        ], dtype=np.uint8)

    # -----------------------------
    # ТЕСТЫ ДЛЯ ImageProcessing
    # -----------------------------

    def test_rgb_to_grayscale(self):
        img = self.sample_color()
        proc = ImageProcessing()
        gray = proc._rgb_to_grayscale(img)

        assert gray.shape == (3, 3)
        assert gray.dtype == np.uint8
        assert gray[0, 0] == round(0.299 * 10 + 0.587 * 20 + 0.114 * 30)

    def test_gamma_correction(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        proc = ImageProcessing()

        out = proc._gamma_correction(img, gamma=2.2)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_custom_edge_detection_runs(self):
        img = self.sample_color()
        proc = ImageProcessing()

        edges = proc.edge_detection(img)
        assert edges.shape == (3, 3)
        assert edges.dtype == np.uint8

    def test_cv_edge_detection_runs(self):
        img = self.sample_color()
        proc = ImageProcessing()

        edges = proc.cv_edge_detection(img)
        assert edges.dtype == np.uint8

    # -----------------------------
    # ТЕСТЫ ДЛЯ CatImage и наследников
    # -----------------------------

    def test_colorcatimage_init(self):
        img = self.sample_color()
        c = ColorCatImage(img, "breed1", "url1")

        assert isinstance(c, CatImage)
        assert c.image.shape == img.shape
        assert c.breed == "breed1"
        assert c.url == "url1"

    def test_graycatimage_init_from_rgb(self):
        img = self.sample_color()
        g = GrayCatImage(img, "breedX", "urlX")

        assert len(g.image.shape) == 2  # должно быть ч/б
        assert g.breed == "breedX"

    def test_graycatimage_init_from_gray(self):
        img = self.sample_gray()
        g = GrayCatImage(img, "breedY", "urlY")

        assert g.image.shape == img.shape

    def test_cat_add_same_size(self):
        c1 = ColorCatImage(self.sample_color(), "b", "u")
        c2 = ColorCatImage(self.sample_color(), "b", "u")

        result = c1 + c2
        assert result.shape == c1.image.shape
        assert result.dtype == np.uint8

    def test_cat_sub_different_sizes(self):
        c1 = ColorCatImage(np.ones((5, 5, 3), dtype=np.uint8) * 100, "b", "u")
        c2 = ColorCatImage(np.ones((3, 3, 3), dtype=np.uint8) * 10, "b", "u")

        result = c1 - c2
        assert result.shape == c1.image.shape
        assert result.dtype == np.uint8

    def test_cat_cv_edge_detection_returns_graycat(self):
        c = ColorCatImage(self.sample_color(), "b", "u")
        e = c.cv_edge_detection()

        assert isinstance(e, GrayCatImage)
        assert len(e.image.shape) == 2

    def test_cat_my_edge_detection_returns_graycat(self):
        c = ColorCatImage(self.sample_color(), "b", "u")
        e = c.my_edge_detection()

        assert isinstance(e, GrayCatImage)
        assert len(e.image.shape) == 2




if __name__ == "__main__":
    unittest.main()
