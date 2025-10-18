import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from implementation import ImageProcessing
import time  # <-- для измерения времени

# Путь к папке с изображениями
images_folder = "test_images"
input_image_path = os.path.join(images_folder, "NotreDameыеу.jpg")

image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError(f"Не удалось загрузить изображение: {input_image_path}")

# Конвертируем BGR -> RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Создаём объект ImageProcessing
processor = ImageProcessing()

# =================== 1. edge_detection ===================
start_time = time.time()
edges = processor.edge_detection(image_rgb)
edge_time = time.time() - start_time

# =================== 2. corner_detection ===================
start_time = time.time()
corners = processor.corner_detection(image_rgb)
corner_time = time.time() - start_time

# =================== Визуализация ===================
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(edges, cmap='gray')
plt.title(f"Edge Detection\nTime: {edge_time:.3f} sec")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(corners)
plt.title(f"Corner Detection\nTime: {corner_time:.3f} sec")
plt.axis("off")

plt.show()
