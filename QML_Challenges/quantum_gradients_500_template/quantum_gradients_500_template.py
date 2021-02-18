#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #

    qnode(params)
    unshifted = dev.state

    def metric_tensor_entry(i, j, unshifted, shift=np.pi/2):
        shifted_plus = params.copy()
        shifted_plus[i] += shift
        shifted_plus[j] += shift
        qnode(shifted_plus)
        forward = dev.state
        ov_forward = abs(np.vdot(unshifted, forward))**2

        shifted_min_plu = params.copy()
        shifted_min_plu[i] -= shift
        shifted_min_plu[j] += shift
        qnode(shifted_min_plu)
        min_plu = dev.state
        ov_min_plu = abs(np.vdot(unshifted, min_plu)) ** 2

        shifted_plu_min = params.copy()
        shifted_plu_min[i] += shift
        shifted_plu_min[j] -= shift
        qnode(shifted_plu_min)
        plu_min = dev.state
        ov_plu_min = abs(np.vdot(unshifted, plu_min)) ** 2

        shifted_minus = params.copy()
        shifted_minus[i] -= shift
        shifted_minus[j] -= shift
        qnode(shifted_minus)
        backward = dev.state
        ov_backward = abs(np.vdot(unshifted, backward)) ** 2

        me = (1/8) * (-ov_forward + ov_plu_min + ov_min_plu - ov_backward)

        return me

    def apply_parameter_shift(qnode, params, i):
        shifted = params.copy()
        shifted[i] += np.pi / 2
        forward = qnode(shifted)  # forward evaluation

        shifted[i] -= np.pi
        backward = qnode(shifted)  # backward evaluation

        return 0.5 * (forward - backward)

    gradient = np.zeros([6], dtype=np.float64)
    metric_tensor = np.zeros([6, 6], dtype=np.float64)
    for i in range(6):
        gradient[i] = apply_parameter_shift(qnode, params, i)
        for j in range(6):
            metric_tensor[i][j] = metric_tensor_entry(i, j, unshifted)

    me_inv = np.linalg.inv(metric_tensor)
    natural_grad = me_inv @ gradient
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")

