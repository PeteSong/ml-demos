import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier

data = datasets.load_wine(as_frame=True)

X = data.data
y = data.target

RANDOM_STATE = 22

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
dtree.fit(X_train, y_train)

print(f'Train data accuracy: {accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train))}')
print(f'Test data accuracy: {accuracy_score(y_true=y_test, y_pred=dtree.predict(X_test))}')

estimator_range = [2, 4, 5, 8, 10, 12, 14, 16]
models = []
scores = []
for estimator in estimator_range:
    clf = BaggingClassifier(n_estimators=estimator, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    models.append(clf)
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

clf = BaggingClassifier(n_estimators=12, oob_score=True, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# plt.figure(figsize=(9, 6))
# plt.plot(estimator_range, scores)
# plt.xlabel('n_estimators', {'fontsize':18})
# plt.ylabel('score', {'fontsize':18})
# plt.tick_params(labelsize=16)

fig, axs = plt.subplots(1, 3)
fig.set_size_inches(40, 8)

# draw a decision tree of DecisionTreeClassifier on the subplot 0
plot_tree(dtree, feature_names=X.columns, ax=axs[0], filled=True, fontsize=8)

# draw a plot of accuracy score on the subplot 1
axs[1].plot(estimator_range, scores)
# axs[1].set_xlabel('n_estimators', fontsize=10)
axs[1].set_xlabel('n_estimators')
axs[1].set_ylabel('score')
# axs[1].tick_params(labelsize=9)

# draw a decision tree from the BaggingClassifier on the subplot 2
plot_tree(clf.estimators_[0], feature_names=X.columns, ax=axs[2], filled=True)

plt.show()
