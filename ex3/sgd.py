#################################
# Your name: Tomer Fisher
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.special import softmax

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w = np.zeros(np.size(data, 1))
    for t in range(T):
        index = np.random.randint(0, high=np.size(data, 0))
        eta_t = eta_0 / (t + 1)
        if labels[index] * np.inner(w, data[index]) < 1:
            w = (1 - eta_t) * w + eta_t * C * labels[index] * data[index]
        else:
            w = (1 - eta_t) * w
    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w = np.zeros(np.size(data, 1))
    for t in range(T):
        index = np.random.randint(0, high=np.size(data, 0))
        eta_t = eta_0 / (t + 1)
        exp = softmax(-labels[index] * np.inner(w, data[index]))
        w = w + (eta_t * labels[index] * exp / (1 + exp)) * data[index]
    return w

#################################

# Place for additional code

def SGD_log_all_w(data, labels, eta_0, T):
    w = np.zeros((1, data.shape[1]))
    for t in range(T):
        #print(t)
        index = np.random.randint(0, high=np.size(data, 0))
        eta_t = eta_0 / (t + 1)
        exp = softmax(-labels[index] * np.inner(w[t], data[index]))
        w = np.concatenate((w, np.array([w[t] + (eta_t * exp / (1 + exp)) * data[index]])))
    return w
    

def calc_accuracy(w, data, labels):
    loss = 0
    for x, y in zip(data, labels):
        y_pred = 1 if np.inner(w, x) >=0 else -1
        if y_pred != y:
            loss += 1
    accuracy = 1 - loss / np.size(data, 0)
    return accuracy


def find_best_eta_0_hinge(train_data, train_labels, validation_data, validation_labels, C, eta_0_range, T):
    avg_accuracies = np.empty(0)
    for eta_0 in eta_0_range:
        accuracies = np.empty(0)
        for _ in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            accuracies = np.append(accuracies, calc_accuracy(w, validation_data, validation_labels))
        avg_accuracies = np.append(avg_accuracies, np.average(accuracies))
    plt.figure(1)
    plt.xlabel("η0")
    plt.xscale("log")
    plt.ylabel("accuracy")
    plt.xlim([eta_0_range[0], eta_0_range[-1]])
    plt.plot(eta_0_range, avg_accuracies, marker='o')
    plt.show()
    return eta_0_range[np.argmax(avg_accuracies)]
    
    
def find_best_C_hinge(train_data, train_labels, validation_data, validation_labels, C_range, eta_0, T):
    avg_accuracies = np.empty(0)
    for C in C_range:
        accuracies = np.empty(0)
        for _ in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            accuracies = np.append(accuracies, calc_accuracy(w, validation_data, validation_labels))
        avg_accuracies = np.append(avg_accuracies, np.average(accuracies))
    plt.figure(1)
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("accuracy")
    plt.xlim([C_range[0], C_range[-1]])
    plt.plot(C_range, avg_accuracies, marker='o')
    plt.show()
    return C_range[np.argmax(avg_accuracies)]
    
    
def find_best_eta_0_log(train_data, train_labels, validation_data, validation_labels, eta_0_range, T):
    avg_accuracies = np.empty(0)
    for eta_0 in eta_0_range:
        accuracies = np.empty(0)
        for _ in range(10):
            w = SGD_log(train_data, train_labels, eta_0, T)
            accuracies = np.append(accuracies, calc_accuracy(w, validation_data, validation_labels))
        avg_accuracies = np.append(avg_accuracies, np.average(accuracies))
    plt.figure(2)
    plt.xlabel("η0")
    plt.xscale("log")
    plt.ylabel("accuracy")
    plt.xlim([eta_0_range[0], eta_0_range[-1]])
    plt.plot(eta_0_range, avg_accuracies, marker='o')
    plt.show()
    return eta_0_range[np.argmax(avg_accuracies)]


def plot_norm_vs_iterations(train_data, train_labels, eta_0, T):
    all_ws = SGD_log_all_w(train_data, train_labels, eta_0, T)
    ws_norm = [np.linalg.norm(w) for w in all_ws[1:]]
    plt.figure(5)
    plt.xlabel("t")
    plt.xscale("log")
    plt.ylabel("norm of w_t")
    plt.xlim([1, T])
    plt.plot(range(1, T+1), ws_norm)
    plt.show()
    
    
#################################


if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    log_scale = np.geomspace(10 ** -5, 10 ** 5, num=11)
    
    eta_0 = find_best_eta_0_hinge(train_data, train_labels, validation_data, validation_labels, 1, log_scale, 1000)
    C = find_best_C_hinge(train_data, train_labels, validation_data, validation_labels, log_scale, eta_0, 1000)
    w = SGD_hinge(train_data, train_labels, C, eta_0, 20000)
    plt.figure(3)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    print("The accuracy of the best classifier on the test set is " + str(calc_accuracy(w, test_data, test_labels)))
    
    eta_0 = find_best_eta_0_log(train_data, train_labels, validation_data, validation_labels, log_scale, 1000)
    w = SGD_log(train_data, train_labels, eta_0, 20000)
    plt.figure(4)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    print("The accuracy of the best classifier on the test set is " + str(calc_accuracy(w, test_data, test_labels)))
    
    plot_norm_vs_iterations(train_data, train_labels, eta_0, 20000)
