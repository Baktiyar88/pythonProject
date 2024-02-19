import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def knn_distance(point, candidates, k):
    distances = cdist([point], candidates, 'euclidean')[0]
    return np.partition(distances, k)[:k]

def reachability_distance(k_distance_x, x, y):
    return max(k_distance_x, np.linalg.norm(x - y))

def local_reachability_density(k, X, knn_distances):
    lrd = []
    for i, x in enumerate(X):
        reach_dists = [reachability_distance(knn_distances[j][-1], x, X[int(j)]) for j in knn_distances[i]]
        lrd.append(k / np.sum(reach_dists))
    return np.array(lrd)

def lof(X, k):
    knn_distances = [np.argsort(cdist([x], X, 'euclidean')[0])[1:k+1] for x in X]
    lrd_values = local_reachability_density(k, X, knn_distances)
    lof_scores = [np.mean([lrd_values[int(j)] / lrd_values[i] for j in knn_distances[i]]) for i, x in enumerate(X)]
    return np.array(lof_scores)

if __name__ == "__main__":
    # Загрузка данных Iris
    data = load_iris()
    X = data.data

    # Расчет LOF с нашей реализацией
    lof_scores = lof(X, k=5)

    # Определение порога для выбросов
    threshold = 1.5

    # Определение выбросов на основе порога
    outliers_indices = np.where(lof_scores > threshold)[0]
    print("Индексы выбросов:", outliers_indices)

    # Использование LOF из scikit-learn для сравнения
    clf = LocalOutlierFactor(n_neighbors=5)
    clf.fit(X)
    sklearn_lof_scores = -clf.negative_outlier_factor_
    sklearn_outliers_indices = np.where(sklearn_lof_scores > threshold)[0]
    print("Индексы выбросов scikit-learn:", sklearn_outliers_indices)

    # Построение диаграммы рассеяния для первых двух признаков
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c='b', label='Data points')
    plt.scatter(X[outliers_indices, 0], X[outliers_indices, 1], c='r', marker='o', edgecolors='k', label='Пользовательские выбросы LOF')
    plt.scatter(X[sklearn_outliers_indices, 0], X[sklearn_outliers_indices, 1], c='g', marker='x', label='Scikit-learn LOF выбросы')
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title('Локальный фактор выброса (LOF)')
    plt.legend()
    plt.show()
