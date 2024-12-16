#!/bin/python3

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# Функция для вычисления расстояний от точек до центров кластеров
def compute_distance(X, centroids):
    dist = torch.cdist(X, centroids, p=2)
    return dist


# Реализация метода K-средних
def k_means(X, k, max_iters=100, tolerance=1e-4, device='cuda'):
    # Переводим все данные на нужное устройство (GPU)
    X = X.to(device)

    # Инициализация центров случайным образом
    centroids = X[torch.randint(0, X.size(0), (k,))].to(device)

    for i in range(max_iters):
        # Вычисление расстояний от всех точек до центров
        dist = compute_distance(X, centroids)

        # Присваиваем каждой точке ближайший центр
        labels = dist.argmin(dim=1)

        # Пересчитываем центры кластеров
        new_centroids = torch.stack([X[labels == j].mean(dim=0) for j in range(k)], dim=0)

        # Проверяем сходимость (если центры не изменились)
        centroid_shift = ((new_centroids - centroids) ** 2).sum().sqrt().item()
        if centroid_shift < tolerance:
            print(f"Convergence reached at iteration {i + 1}")
            break

        centroids = new_centroids

        # Отладочная информация
        print(f"Iteration {i + 1}/{max_iters}, Centroid shift: {centroid_shift}")

    return centroids, labels


# Функция для вычисления внутрикластерной суммы квадратов (для метода локтя)
def calculate_inertia(X, k, device='cuda'):
    centroids, labels = k_means(X, k, max_iters=100, tolerance=1e-4, device=device)
    dist = compute_distance(X, centroids)
    inertia = dist.min(dim=1).values.sum().item()  # Сумма квадратов расстояний от точек до центров
    return inertia


# Функция для построения графика локтя
def plot_elbow(X, max_k=10, device='cuda'):
    inertias = []
    for k in range(1, max_k + 1):
        inertia = calculate_inertia(X, k, device)
        inertias.append(inertia)
        print(f"K={k}, Inertia={inertia}")
    
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

    # Возвращаем k, где инерция начинает стабилизироваться
    optimal_k = np.argmin(np.diff(inertias)) + 2  # +2, потому что diff уменьшается на 1
    print(f"Suggested optimal k based on the Elbow method: {optimal_k}")
    return optimal_k


# Функция для визуализации результатов кластеризации
def plot_clusters(X, centroids, labels, device='cuda'):
    X_cpu = X.cpu()
    labels_cpu = labels.cpu()
    centroids_cpu = centroids.cpu()

    plt.scatter(X_cpu[:, 0].numpy(), X_cpu[:, 1].numpy(), c=labels_cpu.numpy(), cmap='viridis', s=50, alpha=0.6)
    plt.scatter(centroids_cpu[:, 0].numpy(), centroids_cpu[:, 1].numpy(), c='red', marker='x', s=200)
    plt.title("K-Means Clustering")
    plt.show()


# Функция для многократных запусков K-средних
def k_means_multiple_runs(X, k=3, max_runs=10, max_iters=100, tolerance=1e-4, device='cuda'):
    best_centroids = None
    best_labels = None
    best_inertia = float('inf')

    for _ in range(max_runs):
        centroids, labels = k_means(X, k, max_iters, tolerance, device)
        dist = compute_distance(X, centroids)
        inertia = dist.min(dim=1).values.sum().item()
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels


# Главная функция
if __name__ == '__main__':
    # Чтение данных из CSV файла
    file_path = 'WineQT.csv'  # Укажите путь к вашему файлу
    df = pd.read_csv(file_path)

    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Преобразуем данные в тензор PyTorch
    X = torch.tensor(X_scaled, dtype=torch.float32)

    # Перемещаем данные на GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = X.to(device)

    # Выбираем оптимальное количество кластеров с помощью метода локтя
    print("Using Elbow Method to find optimal k...")
    optimal_k = plot_elbow(X, max_k=10, device=device)

    # Запрашиваем у пользователя выбор количества кластеров (если нужно)
    k_input = input(f"Suggested optimal k is {optimal_k}. Do you want to use this or enter a different value? (Press Enter to use {optimal_k}) ")
    k = int(k_input) if k_input else optimal_k

    # Выполнение кластеризации методом K-средних с выбранным k
    print(f"Running K-Means with k={k}...")
    centroids, labels = k_means(X, k, max_iters=100, tolerance=1e-4, device=device)

    # Визуализация результатов кластеризации
    plot_clusters(X, centroids, labels, device=device)

    # Если нужно выполнить несколько запусков с K-средними
    print("Running multiple K-Means runs...")
    centroids, labels = k_means_multiple_runs(X, k, max_runs=10, max_iters=100, tolerance=1e-4, device=device)

    # Визуализация результатов после нескольких запусков
    plot_clusters(X, centroids, labels, device=device)

    # Выполнение кластеризации методом DBSCAN (если кластеризация K-средними не дала хороших результатов)
    print("Running DBSCAN...")
    db = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = db.fit_predict(X_scaled)

    # Визуализация кластеров DBSCAN
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_dbscan, cmap='viridis', s=50, alpha=0.6)
    plt.title("DBSCAN Clustering")
    plt.show()

