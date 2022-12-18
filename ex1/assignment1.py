import numpy as np
import matplotlib.pyplot as plt

#load MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

#define the training and test set of images
import numpy.random
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def knn_algorithm(train, train_labels, sample, k):
    distances = []
    #calculate the distances
    for index in range(len(train)):
        distances.append(np.linalg.norm(train[index,:] - sample))
    #found the indexes of the k smallest distances
    smallest_distances = np.argpartition(distances, k)
    #get the labels of the k nearest neighbors
    neighbor_labels = np.array(train_labels[smallest_distances[:k]]).astype(int)
    #return the majority label
    return np.argmax(np.bincount(neighbor_labels))

def prediction_accuracy(train, train_labels, test, test_labels, k, n):
    currect_count = 0
    #predict label for each item in the test and compare to the true label
    for index in range(len(test)):
        predict_label = knn_algorithm(train[:n], train_labels[:n], test[index], k)
        if int(predict_label) == int(test_labels[index]):
            currect_count += 1
    #return the prediction accuracy
    return currect_count/len(test)

#print the accuracy of the prediction when the algorithm using 1000 training images and k=10
print(prediction_accuracy(train, train_labels, test, test_labels, 10, 1000))

#plot the prediction accuracy as a function of k, for k=1,...,100 and n=1000
accuracy = []
for k in range(1,101):
    accuracy.append(prediction_accuracy(train, train_labels, test, test_labels, k, 1000))
plt.figure(1)
plt.title("k-NN algorithm with n=1000 training images")
plt.xlabel("k")
plt.ylabel("prediction accuracy")
plt.plot(range(1,101), accuracy)

#plot the prediction accuracy as a function of k, for k=1 and n=100,200,...,5000
accuracy = []
for n in range(100,5001,100):
    accuracy.append(prediction_accuracy(train, train_labels, test, test_labels, 1, n))
plt.figure(2)
plt.title("k-NN algorithm with k=1")
plt.xlabel("n training images")
plt.ylabel("prediction accuracy")
plt.plot(range(100,5001,100), accuracy)
