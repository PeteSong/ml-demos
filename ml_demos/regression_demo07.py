import numpy
from sklearn import datasets, linear_model


def logic2prob(logr, X):
    log_odds = logr.coef_ * X + logr.intercept_
    odds = numpy.exp(log_odds)
    probability = odds / (1 + odds)
    return probability


def model_for_tumor():
    # Reshaped for Logistic function.
    X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
    y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    logr = linear_model.LogisticRegression()
    logr.fit(X, y)

    log_odds = logr.coef_
    odds = numpy.exp(log_odds)
    print(f'Coefficient: {log_odds}, Odds: {odds}')

    predicted = logr.predict(numpy.array([3.46]).reshape(-1, 1))
    print(predicted)

    print(f'Probability: {logic2prob(logr, X)}')


model_for_tumor()


def model_for_iris_with_grid_search():
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    logr = linear_model.LogisticRegression(max_iter=10000)
    C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    scores = []
    for choice in C:
        logr.set_params(C=choice)
        logr.fit(X, y)
        scores.append(logr.score(X, y))
    print(f'\nGrid search => C and score: {list(zip(C, scores))}')


model_for_iris_with_grid_search()
