import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import prepare_unlabeled_dataset


def cluster_images(folder_list):
    # Подготвя снимките без етикети от посочените папки, като ги преоразмерява до 64x64 пиксела и ги нормализира.
    X, names = prepare_unlabeled_dataset(folder_list)

    if len(X) == 0:
        print("Няма намерени снимки за клъстеризация.")
        return

    # Нормализация на данните (важно за K-Means, за да не бъде доминиран от по-големи стойности)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделяне на снимките в 2 групи (може да се промени на 3 или повече, ако има повече категории)
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    print("\nРезултат от клъстеризацията:\n")
    for name, cluster in zip(names, clusters):
        print(f"{name} -> Клъстер {cluster}")

    plot_clusters(X_scaled, names)


def plot_clusters(X_scaled, names):
    plt.figure(figsize=(8, 6))

    for i in range(len(X_scaled)):
        x = X_scaled[i, 0]
        y = X_scaled[i, 1]
        plt.scatter(x, y)
        plt.text(x + 0.03, y + 0.03, names[i], fontsize=8)

    plt.title("Клъстеризация на изображения")
    plt.xlabel("Характеристика 1")
    plt.ylabel("Характеристика 2")
    plt.grid(True)
    plt.show()