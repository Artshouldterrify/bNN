import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt

# class for each layer
class layer:
    def __init__(self, num_neurons, num_next, alpha):
        self.size = num_neurons
        self.next_size = num_next
        self.bias = list()
        for i in range(num_neurons):
            self.bias.append(rand.randint(1, 100)/100)
        self.weights = list()
        for i in range(num_neurons):
            neuron_weights = list()
            for j in range(num_next):
                neuron_weights.append(rand.randint(1, 100)/100)
            self.weights.append(neuron_weights)
        self.input = list()
        self.outputs = list()
        self.alpha = alpha

    def display(self):
        print("Layer--")
        for i in range(self.size):
            print("Neuron ", i + 1, ":", self.weights[i])

    def get_size(self):
        return self.size

    def get_weight(self, i, j):
        return self.weights[i][j]

    @staticmethod
    def derivative(x):
        return x * (1 - x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))


# hidden layer: layer
class hidden_layer(layer):
    def __init__(self, num_neurons, num_next, alpha):
        super().__init__(num_neurons, num_next, alpha)
        self.error_vector = list()

    def forward_prop(self, inputs):
        self.input = inputs
        for i in range(self.size):
            self.input[i] += self.bias[i]
        self.outputs = self.sigmoid(np.array(self.input))
        inputs_next = list()
        for i in range(self.next_size):
            inputs_next.append(0)
        for i in range(self.size):
            for j in range(self.next_size):
                inputs_next[j] += self.outputs[i] * self.weights[i][j]
        return inputs_next

    def backpropogation(self, delta_vector):
        delta_net = list()
        for i in range(self.size):
            delta_net.append(0)
        for i in range(self.size):
            delta_net[i] = np.dot(delta_vector, self.weights[i])
        delta_net = np.multiply(delta_net, self.derivative(self.outputs))
        self.error_vector = delta_vector
        return delta_net

    def update_weights(self):
        for i in range(self.size):
            self.weights[i] = np.subtract(self.weights[i], np.multiply(self.error_vector, self.input[i]) * self.alpha)


# class input layer: layer
class input_layer(layer):
    def __init__(self, num_neurons, num_next, alpha):
        super().__init__(num_neurons, num_next, alpha)

    def forward_prop(self, inputs):
        self.input = inputs
        self.outputs = self.input
        inputs_next = list()
        for i in range(self.next_size):
            inputs_next.append(0)
        for i in range(self.size):
            for j in range(self.next_size):
                inputs_next[j] += self.outputs[i] * self.weights[i][j]
        return inputs_next

    def end_backprop(self, delta_vector):
        for i in range(self.size):
            self.weights[i] = np.subtract(self.weights[i], np.multiply(delta_vector, self.input[i]) * self.alpha)

    def adaline_update(self, error, t_in):
        for i in range(self.size):
            self.weights[i] = np.add(self.weights[i], self.alpha * error * t_in[i])
        return

    def perceptron_update(self, net_output, t_in):
        for i in range(self.size):
            self.weights[i][0] += self.alpha * net_output * t_in[i]
        print("UP")
        return


# class output layer: layer
class output_layer(layer):
    def __init__(self, num_neurons, num_next, alpha):
        super().__init__(num_neurons, num_next, alpha)
        self.output_vector = list()
        self.error_vector = list()
        for i in range(self.size):
            self.error_vector.append(0)

    def assign_outputs(self, out_vector):
        self.output_vector = out_vector

    def end_forward(self, inputs):
        self.input = inputs
        for i in range(self.size):
            self.input[i] += self.bias[i]
        self.outputs = self.sigmoid(np.array(self.input))

    def pick_output_class(self):
        picked_class = -1
        max_value = 0
        for i in range(self.size):
            if self.outputs[i] > max_value:
                max_value = self.outputs[i]
                picked_class = i
        return picked_class

    def error(self):
        err = 0.0
        for i in range(self.size):
            err += abs(self.outputs[i] - self.output_vector[i])
        return err

    def backprop_error(self):
        for i in range(self.size):
            self.error_vector[i] = (self.outputs[i] - self.output_vector[i]) * self.derivative(self.outputs[i])
        return self.error_vector




class BackpropogationNetwork:
    """
    Implements a backpropogation network with changeable numbers of layers and hidden neurons.
    """
    def __init__(self, num_hid, hid_n, inp_n, out_n, alpha):
        """
        Initialize network.
        :param num_hid: number of hidden neuron layers, int.
        :param hid_n: number of hidden neurons, int.
        :param inp_n: number of input neurons, int.
        :param out_n: number of output neurons equal to the number of output classes, int.
        :param alpha: training rate, float.
        """
        self.__arr = list()
        input_l = input_layer(inp_n, hid_n, alpha)
        self.__arr.append(input_l)
        for i in range(num_hid - 1):
            temp = hidden_layer(hid_n, hid_n, alpha)
            self.__arr.append(temp)
        temp = hidden_layer(hid_n, out_n, alpha)
        self.__arr.append(temp)
        out_l = output_layer(out_n, out_n, alpha)
        self.__arr.append(out_l)

    def get_arr(self):
        """
        Returns the network as a list of layer object.
        :return: (list)
        """
        return self.__arr

    def display_network(self):
        """
        Display current state of network.
        """
        for i in range(len(self.__arr) - 1):
            self.__arr[i].display()
        print("---------------------------------------------------------------")
        return

    def __forward_propogation_phase(self, ds_inputs):
        """
        Private func for forward propogation.
        :param ds_inputs: Current inputs.
        :return: nothing.
        """
        current = self.__arr[0].forward_prop(ds_inputs)
        for i in range(1, len(self.__arr) - 1):
            current = self.__arr[i].forward_prop(current)
        self.__arr[len(self.__arr) - 1].end_forward(current)
        return

    def __backpropogation_phase(self, correct_vector):
        """
        Private func for backward propogation.
        :param correct_vector: output list.
        :return: nothing.
        """
        self.__arr[len(self.__arr) - 1].assign_outputs(correct_vector)
        current = self.__arr[len(self.__arr) - 1].backprop_error()
        for i in range(len(self.__arr) - 2):
            current = self.__arr[len(self.__arr) - 2 - i].backpropogation(current)
        self.__arr[0].end_backprop(current)
        for i in range(1, len(self.__arr) - 1):
            self.__arr[i].update_weights()
        return

    def train_network(self, training_dataset, epoch_num, outs):
        """
        Training the network.
        :param training_dataset: normalized training dataset, pandas dataframe. Should only contain features.
        :param epoch_num: number of epochs to train the network for, int.
        :param outs: the output vector for the training dataset. Use method get_out_vector to generate this.
        :return: nothing.
        """
        for epoch in range(epoch_num):
            for i in range(len(training_dataset)):
                tup = training_dataset.iloc[i].tolist()
                self.__forward_propogation_phase(tup)
                self.__backpropogation_phase(outs[i])
        return

    def test_network(self, testing_dataset, outs):
        """
        Test the network against a normalized testing dataset.
        :param testing_dataset: the testing dataset, pandas dataframe. Should only contain features.
        :param outs: the output vector for the testing dataset. Use method get_out_vector to generate this.
        :return: accuracy (float), a list of predicted outputs for the dataset (list)
        """
        correct = 0
        predict_arr = list()
        for i in range(len(testing_dataset)):
            tup = testing_dataset.iloc[i].tolist()
            self.__forward_propogation_phase(tup)
            out_class = self.__arr[len(self.__arr) - 1].pick_output_class()
            predict_arr.append(out_class)
            if outs[i][out_class] == 1:
                correct += 1
        accuracy = (correct / len(testing_dataset)) * 100
        return accuracy, predict_arr

    @staticmethod
    def get_out_vector(outputs_ds, num_outs):
        """
        Utility function to generate an output vector.
        :param outputs_ds: a list containing the correct output class for each training tuple.
        :param num_outs: number of output classes.
        :return: output vector (list)
        """
        out_vector = list()
        for i in range(len(outputs_ds)):
            temp = [0 for x in range(num_outs)]
            temp[outputs_ds[i]] = 1
            out_vector.append(temp)
        return out_vector


class AdalineNetwork:
    """
    Implements a simple Adaptive Linear Neuron with bipolar inputs and outputs.
    """
    def __init__(self, num_inp, alpha):
        """
        Initialize the network.
        :param num_inp: number of input features/neurons, int.
        :param alpha: training rate, float.
        """
        self.__input_l = input_layer(num_inp, 1, alpha)
        self.__output_l = output_layer(1, 1, alpha)
        self.__arr = list()
        self.__arr.append(self.__input_l)
        self.__arr.append(self.__output_l)
        self.__out_bias = rand.randint(1, 100)/100
        self.alpha = alpha

    def get_arr(self):
        """
        Returns the network as a list of layer object.
        :return: (list)
        """
        return self.__arr

    def display_network(self):
        """
        Display current state of network.
        """
        self.__input_l.display()
        print("---------------------------------------------------------------")
        return

    def train_network(self, training_set, training_outputs, epochs):
        """
        Train the network.
        :param training_set: training dataset, pandas dataframe. Should only contain features.
        :param training_outputs: list of output labels for each training tuple, list.
        :param epochs: number of epochs for which the network is to be trained, int.
        :return: nothing.
        """
        for e in range(epochs):
            for t_i in range(len(training_set)):
                inp = self.__input_l.forward_prop(training_set.iloc[t_i].tolist())
                net_input = inp[0] + self.__out_bias
                out = -1.0
                if net_input >= 0.0:
                    out = 1.0
                error = training_outputs[t_i] - out
                self.__input_l.adaline_update(error, training_set.iloc[t_i].tolist())
                self.__out_bias += self.alpha * error
        return

    def test_network(self, testing_set, testing_outputs):
        """
        Test the network.
        :param testing_set: testing dataset, pandas dataframe. Should only contain features.
        :param testing_outputs: list of output labels for each training tuple, list.
        :return: accuracy (int), output labels for each tuple (list)
        """
        correct = 0
        predict_arr = list()
        for i in range(len(testing_set)):
            inp = self.__input_l.forward_prop(testing_set.iloc[i].tolist())
            net_input = inp[0] + self.__out_bias
            out = -1.0
            if net_input >= 0.0:
                out = 1.0
            if testing_outputs[i] == out:
                correct += 1
            predict_arr.append(out)
        accuracy = correct/len(testing_set) * 100
        return accuracy, predict_arr




class PerceptronNetwork:
    """
    Implements a simple perceptron network with bipolar inputs and outputs.
    """
    def __init__(self, num_inp, alpha):
        """
        Initialize the network.
        :param num_inp: number of input features/neurons, int.
        :param alpha: training rate, float.
        """
        self.__input_l = input_layer(num_inp, 1, alpha)
        self.__output_l = output_layer(1, 1, alpha)
        self.__arr = list()
        self.__arr.append(self.__input_l)
        self.__arr.append(self.__output_l)
        self.__out_bias = rand.randint(1, 100)/100
        self.alpha = alpha

    def get_arr(self):
        """
        Returns the network as a list of layer object.
        :return: (list)
        """
        return self.__arr

    def display_network(self):
        """
        Display current state of network.
        """
        self.__input_l.display()
        print("---------------------------------------------------------------")
        return

    def train_network(self, training_set, training_outputs, epochs):
        """
        Train the network.
        :param training_set: training dataset, pandas dataframe. Should only contain features.
        :param training_outputs: list of output labels for each training tuple, list.
        :param epochs: number of epochs for which the network is to be trained, int.
        :return: nothing.
        """
        for e in range(epochs):
            for t_i in range(len(training_set)):
                inp = self.__input_l.forward_prop(training_set.iloc[t_i].tolist())
                net_input = inp[0] + self.__out_bias
                out = -1.0
                if net_input >= 0.0:
                    out = 1.0
                if out != training_outputs[t_i]:
                    self.__input_l.perceptron_update(training_outputs[t_i], training_set.iloc[t_i].tolist())
                    self.__out_bias += self.alpha * out
        return

    def test_network(self, testing_set, testing_outputs):
        """
        Test the network.
        :param testing_set: testing dataset, pandas dataframe. Should only contain features.
        :param testing_outputs: list of output labels for each training tuple, list.
        :return: accuracy (int), output labels for each tuple (list)
        """
        correct = 0
        predict_arr = list()
        for i in range(len(testing_set)):
            inp = self.__input_l.forward_prop(testing_set.iloc[i].tolist())
            net_input = inp[0] + self.__out_bias
            out = -1.0
            if net_input >= 0.0:
                out = 1.0
            if testing_outputs[i] == out:
                correct += 1
            predict_arr.append(out)
        accuracy = correct/len(testing_set) * 100
        return accuracy, predict_arr



def normalize(dataset):
    """
    Utility function to normalize a dataset.
    :param dataset: the dataset, pandas dataframe.
    :return: returns the dataset, pandas dataframe.
    """
    for col in dataset.columns[0: len(dataset.iloc[0]) - 1]:
        dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())
    return dataset


def normalize_bipolar(dataset):
    """
    Utility function to normalize a dataset.
    :param dataset: the dataset, pandas dataframe.
    :return: returns the dataset, pandas dataframe.
    """
    for col in dataset.columns[0: len(dataset.iloc[0]) - 1]:
        dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())
        dataset[col] = (dataset[col]*2)
        dataset[col] -= 1
    return dataset


def visualize_network(network):
    """
    Display the network as a graph.
    :param network: the object instance of the network being displayed. A BackpropogationNetwork, AdalineNetwork or
        PerceptronNetwork.
    :return: nothing.
    """
    arr = network.get_arr()
    fig = plt.figure(figsize=(10, 5))
    num_layers = len(arr)
    h_spacing = (900 - (40 * num_layers)) / (num_layers - 1)
    positions = list()
    x_pos, y_pos = 70, 20
    for i in range(len(arr)):
        num_neurons = arr[i].get_size()
        extra_margin = (12 - num_neurons) * 20
        y_pos += extra_margin / 2
        if num_neurons == 1:
            v_spacing = 0
        else:
            v_spacing = (500 - extra_margin - (40 * num_neurons)) / (num_neurons - 1)
        temp = list()
        for j in range(num_neurons):
            plt.scatter(x_pos, y_pos, s=400, c="black")
            temp.append((x_pos, y_pos))
            y_pos += 40 + v_spacing
        positions.append(temp)
        y_pos = 20
        x_pos += 40 + h_spacing
    plt.title("Conceptual Structure of Network")
    for i in range(num_layers - 1):
        size_A = arr[i].get_size()
        size_B = arr[i + 1].get_size()
        for j in range(size_A):
            for k in range(size_B):
                x_vals = [positions[i][j][0], positions[i + 1][k][0]]
                y_vals = [positions[i][j][1], positions[i + 1][k][1]]
                plt.plot(x_vals, y_vals, c="grey")
    plt.show()
