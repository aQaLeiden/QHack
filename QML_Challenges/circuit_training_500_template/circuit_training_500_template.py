#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def qcircuit(params, x, depth=1):
        params = list(params)
        for layer in range(depth):
            qml.Rot(*x, wires=0)
            qml.Rot(*x, wires=1)
            qml.Rot(*x, wires=2)

            qml.Rot(params.pop(), params.pop(), params.pop(), wires=0)
            qml.Rot(params.pop(), params.pop(), params.pop(), wires=1)
            qml.Rot(params.pop(), params.pop(), params.pop(), wires=2)

        return [qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1)), qml.expval(qml.PauliZ(wires=2))]

    def cost(params, x, y, depth):
        batch_loss = []
        label_vecs = {
            1: [1, 0, 0],
            0: [0, 1, 0],
            -1: [0, 0, 1]
        }

        for i in range(len(x)):
            f = qcircuit(params, x[i], depth=depth)
            label = label_vecs[y[i]]

            s = 0
            for e in range(3):
                s += abs(f[e] - label[e])**2

            batch_loss.append(s)

        m = 0
        for s in batch_loss:
            m += s

        return m / len(x)

    def iterate_minibatches(inputs, targets, batch_size):
        """
        A generator for batches of the input data

        Args:
            inputs (array[float]): input data
            targets (array[float]): targets

        Returns:
            inputs (array[float]): one batch of input data of length `batch_size`
            targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

    num_layers = 2
    learning_rate = 0.1
    epochs = 10
    batch_size = 10

    opt = qml.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

    # initialize random weights
    params = [np.random.uniform(0, np.pi) for _ in range(3*3*num_layers)]
    # params = [1.9107789169579406, -1.3361010880402893, 2.5729120915865447, 0.5462620353728947, 1.8639246247816168, 2.82881385254123, 2.9787306016024866, 0.27762877750857795, 2.9322884562491978, 2.554849451249985, 1.46259739909039, 0.16547827889135963, 0.9132772818632782, 1.5778793241001374, 1.8450191507666864, 0.9810311625079425, 1.746634979354536, 1.8117798508230185]


    for it in range(epochs):
        for Xbatch, ybatch in iterate_minibatches(X_train, Y_train, batch_size=batch_size):
            params = opt.step(lambda v: cost(v, Xbatch, ybatch, num_layers), params)

        # params = opt.step(lambda v: cost(v, X_train, Y_train, num_layers), params)
        # loss = np.mean(cost(params, X_train, Y_train, num_layers))

        # res = [it + 1, loss]
        # print(
        #     "Epoch: {:2d} | Loss: {:3f}".format(
        #         *res
        #     )
        # )

    predictions = []
    label_choices = {0: 1, 1: 0, 2: -1}
    for x in X_test:
        pred = qcircuit(params, x, depth=num_layers)
        label = label_choices[np.argmax(pred)]
        predictions.append(label)

    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
