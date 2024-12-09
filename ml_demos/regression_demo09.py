from sklearn import datasets
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=k_folds)
print(
    f'''
Do K-Fold cross validation again DecisionTreeClassifier =>
    Cross validation scores: {scores}
    Average CV score: {scores.mean()}
    Number of CV scores: {len(scores)}
'''
)

sk_folds = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=sk_folds)
print(
    f'''
Do Stratified K-Fold cross validation again DecisionTreeClassifier =>
    Cross validation scores: {scores}
    Average CV score: {scores.mean()}
    Number of CV scores: {len(scores)}
'''
)

loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=loo)
print(
    f'''
Do Leave-One-Out cross validation again DecisionTreeClassifier =>
    Cross validation scores: {scores}
    Average CV score: {scores.mean()}
    Number of CV scores: {len(scores)}
'''
)

lpo = LeavePOut(p=2)
scores = cross_val_score(clf, X, y, cv=lpo)
print(
    f'''
Do Leave-P-Out cross validation again DecisionTreeClassifier =>
    Cross validation scores: {scores}
    Average CV score: {scores.mean()}
    Number of CV scores: {len(scores)}
'''
)

ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
scores = cross_val_score(clf, X, y, cv=ss)
print(
    f'''
Do Shuffle Split cross validation again DecisionTreeClassifier =>
    Cross validation scores: {scores}
    Average CV score: {scores.mean()}
    Number of CV scores: {len(scores)}
'''
)
