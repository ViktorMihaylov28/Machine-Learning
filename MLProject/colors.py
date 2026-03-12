import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import load_image


def detect_main_colors(image_path, k=5):
    # Зарежда снимката и я преобразува в RGB формат (ако е необходимо)
    image = load_image(image_path)

    if image is None:
        return

    # Изображението става на списък от пиксели (всяка пиксел е RGB стойност)
    pixels = image.reshape((-1, 3))

    # K-Means разделя пикселите на 5 групи (може да се промени на повече или по-малко, ако искаш) и намира центровете на тези групи, които са основните цветове в снимката.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)

    # Подрежда цветовете по честота (най-често срещаният цвят първи)
    sorted_indices = np.argsort(counts)[::-1]
    colors = colors[sorted_indices]
    counts = counts[sorted_indices]

    print("\nНамерени основни цветове:")
    for i, color in enumerate(colors):
        print(f"Цвят {i + 1}: RGB = {tuple(color)} | Брой пиксели: {counts[i]}")

    show_colors(image, colors, counts)


def show_colors(image, colors, counts):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].imshow(image)
    axes[0].set_title("Оригинална снимка")
    axes[0].axis("off")

    bar = np.zeros((100, 500, 3), dtype="uint8")
    start_x = 0
    total = np.sum(counts)

    for color, count in zip(colors, counts):
        end_x = start_x + int((count / total) * 500)
        bar[:, start_x:end_x, :] = color
        start_x = end_x

    axes[1].imshow(bar)
    axes[1].set_title("5 основни цвята")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()