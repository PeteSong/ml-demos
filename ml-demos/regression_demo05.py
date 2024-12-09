import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

# Draw samples from a binomial distribution.
# 从 二项式分布 中抽取样本
actual = numpy.random.binomial(1, 0.9, size=1000)
predicted = numpy.random.binomial(1, 0.9, size=1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

print(
    f'''
Metrics from the Confusion Matrix:
    Accuracy: {metrics.accuracy_score(actual, predicted)}
    Precision: {metrics.precision_score(actual, predicted)}
    Sensitivity(Recall): {metrics.recall_score(actual, predicted)}
    Specificity: {metrics.recall_score(actual, predicted, pos_label=0)}
    F-score: {metrics.f1_score(actual, predicted)}
'''
)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
cm_display.plot()

plt.show()
