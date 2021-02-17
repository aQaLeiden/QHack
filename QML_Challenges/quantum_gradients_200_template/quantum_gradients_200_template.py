#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None, diff_method="parameter-shift")
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    def apply_hessian_parameter_shift(qnode, params, i, j, unshifted, shift=np.pi/4):
        if i != j:
            shifted_plus = params.copy()
            shifted_plus[i] += shift
            shifted_plus[j] += shift
            forward = qnode(shifted_plus)  # forward evaluation

            shifted_min_plu = params.copy()
            shifted_min_plu[i] -= shift
            shifted_min_plu[j] += shift
            min_plu = qnode(shifted_min_plu)

            shifted_plu_min = params.copy()
            shifted_plu_min[i] += shift
            shifted_plu_min[j] -= shift
            plu_min = qnode(shifted_plu_min)

            shifted_minus = params.copy()
            shifted_minus[i] -= shift
            shifted_minus[j] -= shift
            backward = qnode(shifted_minus)  # backward evaluation

            return (forward - min_plu - plu_min + backward) / ((2*np.sin(shift))**2), None
        else:
            shifted_plus = params.copy()
            shifted_plus[i] += np.pi/2
            forward = qnode(shifted_plus)

            shifted_minus = params.copy()
            shifted_minus[i] -= np.pi/2
            backward = qnode(shifted_minus)
            result_hessian = (forward - 2*unshifted + backward) / 2
            result_gradient = (forward - backward) / 2

            return result_hessian, result_gradient

    unshifted = circuit(weights)
    for i in range(5):
        for j in range(5):
            if i <= j:
                hess, grad = apply_hessian_parameter_shift(circuit, weights, i, j, unshifted, shift=np.pi / 4)
                if i == j:
                    gradient[i] = grad

                hessian[i][j] = hess
                hessian[j][i] = hess

    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
