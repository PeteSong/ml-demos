import matplotlib.pyplot as plt
import pandas
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pandas.read_csv('../datasets/comedy_show_data.csv')
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

print(f'predict for the next show: {dtree.predict([[40, 10, 7, 1]])}')

plt.figure(figsize=(9, 6))
plot_tree(dtree, feature_names=features, filled=True)
plt.show()  # if in the jupyter lab, no need this line.
