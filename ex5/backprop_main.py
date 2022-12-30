import backprop_data
import backprop_network
import matplotlib.pyplot as plt


def excersice_b():
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    learning_rate_range = [10 ** x for x in range(-3, 3)]
    training_accuracies = []
    training_losses = []
    test_accuracies = []
    for learning_rate in learning_rate_range:
        net = backprop_network.Network([784, 40, 10])
        net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=learning_rate, test_data=test_data)
        training_accuracies.append(net.one_hot_accuracy(training_data))
        training_losses.append(net.loss(training_data))
        test_accuracies.append(net.one_label_accuracy(test_data))
    # Plot the training accuracy
    plt.figure(1)
    plt.title("Training Accuracy")
    plt.xlabel("learning rate")
    plt.xscale("log")
    plt.ylabel("accuracy")
    plt.xlim([learning_rate_range[0], learning_rate_range[-1]])
    plt.plot(learning_rate_range, training_accuracies, marker='o')
    plt.show()
    # Plot the training loss
    plt.figure(2)
    plt.title("Training Loss")
    plt.xlabel("learning rate")
    plt.xscale("log")
    plt.ylabel("loss")
    plt.xlim([learning_rate_range[0], learning_rate_range[-1]])
    plt.plot(learning_rate_range, training_losses, marker='o')
    plt.show()
    # Plot the test accuracy
    plt.figure(3)
    plt.title("Test Accuracy")
    plt.xlabel("learning rate")
    plt.xscale("log")
    plt.ylabel("accuracy")
    plt.xlim([learning_rate_range[0], learning_rate_range[-1]])
    plt.plot(learning_rate_range, test_accuracies, marker='o')
    plt.show()


def excercise_c():
    training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, 
            test_data=test_data)
    

def excercise_d():
    training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
    net = backprop_network.Network([784, 784, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, 
            test_data=test_data)


excersice_b()
excercise_c()
excercise_d()