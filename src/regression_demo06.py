import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

fig, axs = plt.subplots(1, 2)

axs[0].set_title('Dendrogram')
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data, ax=axs[0])

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
axs[1].set_title('Scatter')
axs[1].scatter(x, y, c=labels)

plt.show()
