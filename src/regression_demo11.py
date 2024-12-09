import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

new_x = 8
new_y = 21
new_point = [(new_x, new_y)]

data = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data, classes)

prediction = knn.predict(new_point)

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(data, classes)
prediction2 = knn2.predict(new_point)

print(f'''
KNN with neighbor=1, prediction is {prediction}
KNN with neighbor=5, prediction is {prediction2}
''')

fig, axs = plt.subplots(1, 3)

axs[0].set_title('Raw')
axs[0].scatter(x, y, c=classes)

axs[1].set_title('KNN - neighbors=1')
axs[1].scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
axs[1].text(x=new_x - 1.7, y=new_y - 0.7, s=f"new point, class: {prediction[0]}")

axs[2].set_title('KNN - neighbors=5')
axs[2].scatter(x + [new_x], y + [new_y], c=classes + [prediction2[0]])
axs[2].text(x=new_x - 1.7, y=new_y - 0.7, s=f"new point, class: {prediction2[0]}")

plt.show()
