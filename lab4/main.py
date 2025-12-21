from implementation.CatImageProcessor import CatImageProcessor, timeit
import cv2
from rich.logging import RichHandler

import logging

import inspect

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)]
)


if __name__ == "__main__":
    processor = CatImageProcessor(save_dir="results")
    # print(inspect.getsource(processor.process_and_save))
    cats = processor.fetch_images(2)
    processor.process_and_save(cats)
    # r = timeit(processor.process_and_save)
    # processor.process_and_save = r



    img = cats[0]

    # --- EDGE DETECTION ---
    edge = img.edge_detection_custom()
    result1 = img + edge
    result2 = img - edge
    cv2.imwrite("sum_edge.png", result1)
    cv2.imwrite("sub_edge.png", result2)
    logging.info("Saved files: sum_edge.png and sub_edge.png")

    # --- CORNER DETECTION ---
    # кастомная реализация
    corners_custom = img.corner_detection_custom()
    result3 = img + corners_custom
    result4 = img - corners_custom
    cv2.imwrite("sum_corner_custom.png", result3)
    cv2.imwrite("sub_corner_custom.png", result4)
    logging.info("Saved files: sum_corner_custom.png and sub_corner_custom.png")