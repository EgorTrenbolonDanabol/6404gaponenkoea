import unittest

import numpy as np

from lab5.implementation.image_processing import ImageProcessing
from lab5.implementation.CatImage import CatImage
from lab5.implementation.ColorCatImage import ColorCatImage
from lab5.implementation.GreyCatImage import GreyCatImage


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

    def test_custom_edge_detection_runs(self):
        img = self.sample_color()
        proc = ImageProcessing()

        edges = proc.edge_detection(img)
        assert edges.shape == (3, 3)


    def test_cv_edge_detection_runs(self):
        img = self.sample_color()
        proc = ImageProcessing()

        edges = proc.library_edge_detection(img)
        assert edges.shape == (3, 3)

    # -----------------------------
    # ТЕСТЫ ДЛЯ CatImage и наследников
    # -----------------------------

    def test_colorcatimage_init(self):
        img = self.sample_color()
        c = ColorCatImage(img, "breed1", "url1")

        assert isinstance(c, CatImage)

    def test_graycatimage_init_from_rgb(self):
        img = self.sample_color()
        g = GreyCatImage(img, "breedX", "urlX")

        assert len(g.image.shape) == 2  # должно быть ч/б

    def test_graycatimage_init_from_gray(self):
        img = self.sample_gray()
        g = GreyCatImage(img, "breedY", "urlY")

        assert g.image.shape == img.shape

    def test_cat_add_same_size(self):
        c1 = ColorCatImage(self.sample_color(), "b", "u")
        c2 = ColorCatImage(self.sample_color(), "b", "u")

        result = c1 + c2
        assert result.shape == c1.image.shape


    def test_cat_sub_different_sizes(self):
        c1 = ColorCatImage(np.ones((5, 5, 3), dtype=np.uint8) * 100, "b", "u")
        c2 = ColorCatImage(np.ones((3, 3, 3), dtype=np.uint8) * 10, "b", "u")

        result = c1 - c2
        assert result.shape == c1.image.shape


    def test_cat_cv_edge_detection_returns_graycat(self):
        c = ColorCatImage(self.sample_color(), "b", "u")
        e = c.edge_detection_cv()

        assert isinstance(e, GreyCatImage)


    def test_cat_my_edge_detection_returns_graycat(self):
        c = ColorCatImage(self.sample_color(), "b", "u")
        e = c.edge_detection_custom()

        assert isinstance(e, GreyCatImage)


if __name__ == "__main__":
    unittest.main()