import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def initialize_data_labeled_by_circle(n):
    radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
    angles = 2 * math.pi * np.random.random(2 * n)
    X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
    X2 = (radius * np.sin(angles)).reshape((2 * n, 1))
    X = np.concatenate([X1,X2],axis=1)
    y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])
    return X, y


def add_classifier_and_title(classifiers, titles, classifier, title):
    classifiers = np.append(classifiers, classifier)
    titles = np.append(titles, title)
    return classifiers, titles


def excercise_a(X, y):
    classifiers = np.empty(0)
    titles = np.empty(0)
    #add soft SVM with linear kernel
    classifier = svm.SVC(C=10, kernel="linear")
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "linear kernel")
    #add soft SVM with homogeneous polynomial kernel of degree 2
    classifier = svm.SVC(C=10, kernel="poly", degree=2, gamma="auto")
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "poly kernel (d=2)")
    #add soft SVM with homogeneous polynomial kernel of degree 3
    classifier = svm.SVC(C=10, kernel="poly", degree=3, gamma="auto")
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "poly kernel (d=3)")
    plot_results(classifiers, titles, X, y)


def excercise_b(X, y):
    classifiers = np.empty(0)
    titles = np.empty(0)
    #add soft SVM with non-homogeneous polynomial kernel of degree 2
    classifier = svm.SVC(C=10, kernel="poly", degree=2, gamma="auto", coef0=1)
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "poly kernel (d=2)")
    #add soft SVM with non-homogeneous polynomial kernel of degree 3
    classifier = svm.SVC(C=10, kernel="poly", degree=3, gamma="auto", coef0=1)
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "poly kernel (d=3)")
    plot_results(classifiers, titles, X, y)


def perturb_negative_labels(y, probability):
    for index in range(len(y)):
        if y[index] == -1 and np.random.uniform() <= probability:
            y[index] = 1
    return y


def excercise_c(X, y):
    y = perturb_negative_labels(y, 0.1)
    classifiers = np.empty(0)
    titles = np.empty(0)
    #add soft SVM with non-homogeneous polynomial kernel of degree 2
    classifier = svm.SVC(C=10, kernel="poly", degree=2, gamma="auto", coef0=1)
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "poly kernel (d=2)")
    #add soft SVM with rbf kernel with gamma 10
    classifier = svm.SVC(C=10, kernel="rbf", gamma=10)
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "rbf kernel (γ=10)")
    plot_results(classifiers, titles, X, y)
    excercise_c_extension(X, y)


def excercise_c_extension(X, y):
    classifiers = np.empty(0)
    titles = np.empty(0)
    #add soft SVM with rbf kernel with gamma 3
    classifier = svm.SVC(C=10, kernel="rbf", gamma=3)
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "rbf kernel (γ=3)")
    #add soft SVM with rbf kernel with gamma 10
    classifier = svm.SVC(C=10, kernel="rbf", gamma=10)
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "rbf kernel (γ=10)")
    #add soft SVM with rbf kernel with gamma 30
    classifier = svm.SVC(C=10, kernel="rbf", gamma=30)
    classifier.fit(X, y)
    classifiers, titles = add_classifier_and_title(classifiers, titles, classifier, "rbf kernel (γ=30)")
    plot_results(classifiers, titles, X, y)


X, y = initialize_data_labeled_by_circle(100)
excercise_a(X, y)
excercise_b(X, y)
excercise_c(X, y)
