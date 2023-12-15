import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
import time

# 載入資料 (banana資料)
data_banana = pd.read_csv("banana  (with class label).csv")
X_banana = data_banana[['x', 'y']]
y_true_banana = data_banana['class']

# 標準化資料 (banana資料)
scaler_banana = StandardScaler()
X_scaled_banana = scaler_banana.fit_transform(X_banana)

# 設定群數 (banana資料)
n_clusters_banana = 2

# K-means (banana資料)
start_time_banana_kmeans = time.time()
kmeans_banana = KMeans(n_clusters=n_clusters_banana, random_state=42)
y_kmeans_banana = kmeans_banana.fit_predict(X_scaled_banana)
kmeans_time_banana = time.time() - start_time_banana_kmeans

# 計算K-means的SSE (banana資料)
sse_kmeans_banana = np.sum((X_scaled_banana - kmeans_banana.cluster_centers_[y_kmeans_banana])**2)

# 階層式分群 (banana資料)
start_time_banana_agg = time.time()
agg_clustering_banana = AgglomerativeClustering(n_clusters=n_clusters_banana)
y_agg_banana = agg_clustering_banana.fit_predict(X_scaled_banana)
agg_time_banana = time.time() - start_time_banana_agg

# 計算Agglomerative Clustering的SSE (banana資料)
sse_agg_banana = np.sum([np.sum((X_scaled_banana[y_agg_banana == i] - np.mean(X_scaled_banana[y_agg_banana == i], axis=0))**2) for i in range(n_clusters_banana)])

# DBSCAN (banana資料)
start_time_banana_dbscan = time.time()
dbscan_banana = DBSCAN(eps=0.05, min_samples=5)
y_dbscan_banana = dbscan_banana.fit_predict(X_scaled_banana)
dbscan_time_banana = time.time() - start_time_banana_dbscan

# 計算DBSCAN的SSE (banana資料)
sse_dbscan_banana = np.sum((X_scaled_banana - np.mean(X_scaled_banana[y_dbscan_banana != -1], axis=0))**2)

# 顯示結果 (banana資料)
print("\nbanana")
print(f"K-means Time (banana): {kmeans_time_banana:.4f} sec, SSE: {sse_kmeans_banana:.4f}, Accuracy: {accuracy_score(y_true_banana, y_kmeans_banana):.4f}, Entropy: {adjusted_mutual_info_score(y_true_banana, y_kmeans_banana):.4f}")
print(f"Agglomerative Clustering Time (banana): {agg_time_banana:.4f} sec, SSE: {sse_agg_banana:.4f}, Accuracy: {accuracy_score(y_true_banana, y_agg_banana):.4f}, Entropy: {adjusted_mutual_info_score(y_true_banana, y_agg_banana):.4f}")
print(f"DBSCAN Time (banana): {dbscan_time_banana:.4f} sec, SSE: {sse_dbscan_banana:.4f}, Accuracy: {accuracy_score(y_true_banana, y_dbscan_banana):.4f}, Entropy: {adjusted_mutual_info_score(y_true_banana, y_dbscan_banana):.4f}")
print("\n")

# 繪製分群結果 (banana資料)
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(X_scaled_banana[:, 0], X_scaled_banana[:, 1], c=y_kmeans_banana, cmap='viridis', marker='o')
plt.title('K-means (banana)')

plt.subplot(1, 3, 2)
plt.scatter(X_scaled_banana[:, 0], X_scaled_banana[:, 1], c=y_agg_banana, cmap='viridis', marker='o')
plt.title('Agglomerative Clustering (banana)')

plt.subplot(1, 3, 3)
plt.scatter(X_scaled_banana[:, 0], X_scaled_banana[:, 1], c=y_dbscan_banana, cmap='viridis', marker='o')
plt.title('DBSCAN (banana)')

plt.show()

#------------------------------------------------------------------------------------------------------------

# 載入資料 (sizes3資料)
data_sizes3 = pd.read_csv("sizes3 (with class label).csv")
X_sizes3 = data_sizes3[['x', 'y']]
y_true_sizes3 = data_sizes3['class']

# 標準化資料 (sizes3資料)
scaler_sizes3 = StandardScaler()
X_scaled_sizes3 = scaler_sizes3.fit_transform(X_sizes3)

# 設定群數 (sizes3資料)
n_clusters_sizes3 = 4

# K-means (sizes3資料)
start_time_sizes3_kmeans = time.time()
kmeans_sizes3 = KMeans(n_clusters=n_clusters_sizes3, random_state=42)
y_kmeans_sizes3 = kmeans_sizes3.fit_predict(X_scaled_sizes3)
kmeans_time_sizes3 = time.time() - start_time_sizes3_kmeans

# 計算K-means的SSE (sizes3資料)
sse_kmeans_sizes3 = np.sum((X_scaled_sizes3 - kmeans_sizes3.cluster_centers_[y_kmeans_sizes3])**2)

# 階層式分群 (sizes3資料)
start_time_sizes3_agg = time.time()
agg_clustering_sizes3 = AgglomerativeClustering(n_clusters=n_clusters_sizes3)
y_agg_sizes3 = agg_clustering_sizes3.fit_predict(X_scaled_sizes3)
agg_time_sizes3 = time.time() - start_time_sizes3_agg

# 計算Agglomerative Clustering的SSE (sizes3資料)
sse_agg_sizes3 = np.sum([np.sum((X_scaled_sizes3[y_agg_sizes3 == i] - np.mean(X_scaled_sizes3[y_agg_sizes3 == i], axis=0))**2) for i in range(n_clusters_sizes3)])

# DBSCAN (sizes3資料)
start_time_sizes3_dbscan = time.time()
dbscan_sizes3 = DBSCAN(eps=0.3, min_samples=10)
y_dbscan_sizes3 = dbscan_sizes3.fit_predict(X_scaled_sizes3)
dbscan_time_sizes3 = time.time() - start_time_sizes3_dbscan

# 計算DBSCAN的SSE (sizes3資料)
sse_dbscan_sizes3 = np.sum((X_scaled_sizes3 - np.mean(X_scaled_sizes3[y_dbscan_sizes3 != -1], axis=0))**2)

# 顯示結果 (sizes3資料)
print("\nsizes3")
print(f"K-means Time (sizes3): {kmeans_time_sizes3:.4f} sec, SSE: {sse_kmeans_sizes3:.4f}, Accuracy: {accuracy_score(y_true_sizes3, y_kmeans_sizes3):.4f}, Entropy: {adjusted_mutual_info_score(y_true_sizes3, y_kmeans_sizes3):.4f}")
print(f"Agglomerative Clustering Time (sizes3): {agg_time_sizes3:.4f} sec, SSE: {sse_agg_sizes3:.4f}, Accuracy: {accuracy_score(y_true_sizes3, y_agg_sizes3):.4f}, Entropy: {adjusted_mutual_info_score(y_true_sizes3, y_agg_sizes3):.4f}")
print(f"DBSCAN Time (sizes3): {dbscan_time_sizes3:.4f} sec, SSE: {sse_dbscan_sizes3:.4f}, Accuracy: {accuracy_score(y_true_sizes3, y_dbscan_sizes3):.4f}, Entropy: {adjusted_mutual_info_score(y_true_sizes3, y_dbscan_sizes3):.4f}")
print("\n")

# 繪製分群結果 (sizes3資料)
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(X_scaled_sizes3[:, 0], X_scaled_sizes3[:, 1], c=y_kmeans_sizes3, cmap='viridis', marker='o')
plt.title('K-means (sizes3)')

plt.subplot(1, 3, 2)
plt.scatter(X_scaled_sizes3[:, 0], X_scaled_sizes3[:, 1], c=y_agg_sizes3, cmap='viridis', marker='o')
plt.title('Agglomerative Clustering (sizes3)')

plt.subplot(1, 3, 3)
plt.scatter(X_scaled_sizes3[:, 0], X_scaled_sizes3[:, 1], c=y_dbscan_sizes3, cmap='viridis', marker='o')
plt.title('DBSCAN (sizes3)')

plt.show()
