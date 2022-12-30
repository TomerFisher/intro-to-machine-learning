import backprop_data
import backprop_network
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_helper(ylabel, epochs, list_of_yvalues, learning_rates):
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.xlim([epochs[0], epochs[-1]])
    for yvalues, learning_rate, color in zip(list_of_yvalues, learning_rates, mcolors.BASE_COLORS):
        plt.plot(epochs, yvalues, color=color, label='learning rate={0}'.format(learning_rate))
    plt.legend()
    plt.show()


def excersice_b():
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    learning_rates = [10 ** x for x in range(-3, 3)]
    epochs = list(range(1, 31))
    training_accuracies = []
    training_losses = []
    test_accuracies = []
    for learning_rate in learning_rates:
        net = backprop_network.Network([784, 40, 10])
        train_accuracy, train_loss, test_accuracy = net.SGD(training_data, epochs=epochs[-1], mini_batch_size=10, 
                                                            learning_rate=learning_rate, test_data=test_data)
        training_accuracies.append(train_accuracy)
        training_losses.append(train_loss)
        test_accuracies.append(test_accuracy)
    plt.figure(1)
    plot_helper("Training Accuracy", epochs, training_accuracies, learning_rates)
    plt.figure(2)
    plot_helper("Training Loss", epochs, training_losses, learning_rates)
    plt.figure(3)
    plot_helper("Test Accuracy", epochs, test_accuracies, learning_rates)


def train_on_all_data(net_sizes):
    training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
    net = backprop_network.Network(net_sizes)
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, 
            test_data=test_data)


def excercise_c():
    train_on_all_data([784, 40, 10])
    

def excercise_d():
    train_on_all_data([784, 784, 10])


excersice_b()
excercise_c()
excercise_d()