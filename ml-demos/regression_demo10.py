import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

n = 10000

y = np.array([0] * n + [1] * n)

y_prob_1 = np.array(
    np.random.uniform(0.25, 0.5, n // 2).tolist()
    + np.random.uniform(0.3, 0.7, n).tolist()
    + np.random.uniform(0.5, 0.75, n // 2).tolist()
)

y_prob_2 = np.array(
    np.random.uniform(0, 0.4, n // 2).tolist()
    + np.random.uniform(0.3, 0.7, n).tolist()
    + np.random.uniform(0.6, 1, n // 2).tolist()
)

print(
    f'''
model 1 =>
    accuracy score: {accuracy_score(y, y_prob_1 > .5)}
    AUC score: {roc_auc_score(y, y_prob_1)}

model 2 =>
    accuracy score: {accuracy_score(y, y_prob_2 > .5)}
    AUC score: {roc_auc_score(y, y_prob_2)}
'''
)


def plot_roc_curve(ax, title, y_true, y_prob):
    fpr, tpr, threshold = roc_curve(y_true, y_prob)
    ax.set_title(title)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.plot(fpr, tpr)


fig, axs = plt.subplots(1, 2)

plot_roc_curve(axs[0], 'model 1 - ROC curve', y, y_prob_1)
plot_roc_curve(axs[1], 'model 2 - ROC curve', y, y_prob_2)

plt.show()
