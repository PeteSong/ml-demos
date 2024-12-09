import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

fig, axs = plt.subplots(1, 3)

axs[0].set_title('Raw scatter')
axs[0].scatter(x, y)

axs[1].set_title('Elbow method')
axs[1].set_xlabel('Number of clusters')
axs[1].set_ylabel('Inertia')
axs[1].plot(range(1, 11), inertias, marker='o')
# after checking the elbow method graph, 2 is a good value for K

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

axs[2].set_title('Labeled Scatter with K=2')
axs[2].scatter(x, y, c=kmeans.labels_)

plt.show()
