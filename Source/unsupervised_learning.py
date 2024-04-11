from sklearn.mixture import GaussianMixture
from preprocessing_main import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tsne_pca import *


# df_copy = df_normalized.copy()
# df_feature = df_copy.drop('result', axis=1)
labels = df_normalized['result']

#We take only a fixed part of the dataframe
X_1000 = df_pca.iloc[0:1000]
y_1000 = pd.DataFrame(labels.head(1000))

print(tabulate(X_1000.head(), headers='keys', tablefmt='psql'))

#We choose KMeans to perform hard cluster
#Trying to find the best k: we run the algorithm multiple times to evaluate the performance
kmeans_results = []
for k in range(2, 47):
    kmeans_results.append(KMeans(n_clusters=k, init='random').fit(df_feature))

fig, axs = plt.subplots(1,2, figsize=(8, 3.5))

axs[0].plot(
    [k for k in range(2,47)],
    [result.inertia_ for result in kmeans_results]
)
axs[0].set_xlabel("Number of clusters k")
axs[0].set_ylabel("Clustering error")
axs[0].grid()

axs[1].plot(
    [k for k in range(2,47)],
    [silhouette_score(df_feature, result.labels_) for result in kmeans_results]
)
axs[1].set_xlabel("Number of clusters k")
axs[1].set_ylabel("Silhouette")
axs[1].grid()

plt.tight_layout()
plt.show()

#We try to use also the Elbow method
inertia = []
for k in range(2, 47):
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(X_1000)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.plot(range(2, 47), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

#Using KMeans with k=2, as the number of labels and evaluating its performance
kmeans = KMeans(n_clusters=2)
cluster_labels = kmeans.fit_predict(df_feature)

silhouette = silhouette_score(df_feature, cluster_labels)
ri = rand_score(np.ravel(labels), cluster_labels)
ari = adjusted_rand_score(np.ravel(labels), cluster_labels)

print('k-Means with 2 clusters')
(unique, counts)=np.unique( cluster_labels, return_counts=True)
print("Size of each cluster: ", counts)
print(f'k_means clustering error: {round(kmeans.inertia_, 2)}')
print(f'Silhouette: {round(silhouette, 2)}')
print(f'RI: {round(ri, 2)}')
print(f'ARI: {round(ari, 2)}')

#Using KMeans with k=3, as the number of labels and evaluating its performance
kmeans = KMeans(n_clusters=3)
cluster_labels = kmeans.fit_predict(df_feature)

silhouette = silhouette_score(df_feature, cluster_labels)
ri = rand_score(np.ravel(labels), cluster_labels)
ari = adjusted_rand_score(np.ravel(labels), cluster_labels)

print('k-Means with 3 clusters')
(unique, counts)=np.unique( cluster_labels, return_counts=True)
print("Size of each cluster: ", counts)
print(f'k_means clustering error: {round(kmeans.inertia_, 2)}')
print(f'Silhouette: {round(silhouette, 2)}')
print(f'RI: {round(ri, 2)}')
print(f'ARI: {round(ari, 2)}')

#We are using now GMM as a soft clustering algorithm
#We are running multiple times the algorithm in order to find the best hyperparameters
n_cluster_list=[]
shs_list = []
ri_list = []
ari_list = []
log_l_list=[]
for n_clusters in range(3, 16):
    gmm = GaussianMixture(n_components=n_clusters)
    cl_labels_gmm = gmm.fit_predict(df_feature)
    silhouette= silhouette_score(df_feature, cl_labels_gmm)
    n_cluster_list.append(n_clusters)
    shs_list.append(silhouette)
    ri_list.append(rand_score(np.ravel(labels), cl_labels_gmm))
    ari_list.append(adjusted_rand_score(np.ravel(labels), cl_labels_gmm))
    log_l_list.append(gmm.score(df_feature))

best_sh= np.max(shs_list)
best_n=n_cluster_list[np.argmax(shs_list)]
print("best k for GMM: ",best_n, " with corresponding silhouette: ", best_sh)

# Plot
plt.figure(figsize=(5, 3.5))
plt.plot(n_cluster_list,shs_list, marker='o', markersize=5)
plt.scatter(best_n, best_sh, color='r', marker='x', s=90)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()

# Plot GMM total log-likelihood score
plt.figure(figsize=(5, 3.5))
plt.plot(n_cluster_list,log_l_list, marker='o', markersize=5)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('GMM total log-likelihood score')
plt.show()

# Plot ARI
plt.figure(figsize=(5, 3.5))
plt.plot(n_cluster_list,ari_list, marker='o', markersize=5)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('ARI')
plt.show()

#To visualize the result, we use PCA, reducing on the first 2 components
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df_feature), columns=['PCA1', 'PCA2'])

df_visualization = pd.DataFrame({
    'PCA1': df_pca['PCA1'],
    'PCA2': df_pca['PCA2'],
    'GMM_Cluster': cl_labels_gmm,
    'KMeans_Cluster': cluster_labels
})

plt.figure(figsize=(12, 6))
#Plotting GMM
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_visualization, x='PCA1', y='PCA2', hue='GMM_Cluster', palette='viridis')
plt.title('GMM Clustering Results')
#Plotting KMeans
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_visualization, x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='viridis')
plt.title('KMeans Clustering Results')

plt.tight_layout()
plt.show()

#Calculating purity for KMeans
conf_matrix_kmeans = confusion_matrix(labels, cluster_labels)
purity_kmeans = np.sum(np.max(conf_matrix_kmeans, axis=0)) / np.sum(conf_matrix_kmeans)
print('Purity for KMeans with 2 clusters:', round(purity_kmeans, 2))

#Calculating purity for GMM
conf_matrix_gmm = confusion_matrix(labels, cl_labels_gmm)
purity_gmm = np.sum(np.max(conf_matrix_gmm, axis=0)) / np.sum(conf_matrix_gmm)
print('Purity for GMM with best_k clusters:', round(purity_gmm, 2))


