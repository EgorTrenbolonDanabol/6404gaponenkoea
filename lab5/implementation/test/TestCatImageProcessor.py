import unittest
import os


from lab5.implementation.CatImageProcessor import CatImageProcessor
from lab5.implementation.CatImage import CatImage


class TestCatImageProcessorBasic(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_real_results"
        os.makedirs(self.test_dir, exist_ok=True)

        self.processor = CatImageProcessor(save_dir=self.test_dir)

    # ------------------------------------------------------------
    # 1. Реальный тест API (fetch_images)
    # ------------------------------------------------------------
    def test_fetch_images_real_count(self):
        cats = self.processor.fetch_images(limit=2)

        # Проверки
        self.assertEqual(len(cats), 2)

    def test_fetch_images_real_instance(self):
        cats = self.processor.fetch_images(limit=1)

        self.assertIsInstance(cats[0], CatImage)
    # ------------------------------------------------------------
    # 2. Тест записи файлов — process_and_save
    # ------------------------------------------------------------
    def test_process_and_save_real(self):
        cats = self.processor.fetch_images(limit=1)
        self.processor.process_and_save(cats)

        breed_name = cats[0].breed.lower().replace(" ", "_")

        # Пути файлов
        original = os.path.join(self.test_dir, f"1_{breed_name}_original.png")

        # Проверки файлов
        self.assertTrue(os.path.exists(original))


if __name__ == "__main__":
    unittest.main()