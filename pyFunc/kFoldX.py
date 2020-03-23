# k-fold cross val
# Given an estimator, the cross-validation object and the input dataset, the cross_val_score 
# splits the data repeatedly into a training and a testing set, trains the estimator using 
# the training set and computes the scores based on the testing set for each iteration of 
# cross-validation.

from sklearn import datasets, svm
X_digits, y_digits = datasets.load_digits(return_X_y=True)
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])


# To get a better measure of prediction accuracy (which we can use as a proxy for goodness 
# of fit of the model), we can successively split the data in folds that we use for training 
# and testing:


import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)

# perform split

from sklearn.model_selection import KFold, cross_val_score
X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
k_fold = KFold(n_splits=5)
for train_indices, test_indices in k_fold.split(X):
     print('Train: %s | test: %s' % (train_indices, test_indices))


# validate

[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
 for train, test in k_fold.split(X_digits)]

# n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.

cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)



# Alternatively, the scoring argument can be provided to specify an alternative scoring method.


cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro')