#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    def variational_ansatz(params, wires, *, state_n):
        """
        Args:
            params (np.ndarray): An array of floating-point numbers with size (n, 3),
                where n is the number of parameter sets required (this is determined by
                the problem Hamiltonian).
            wires (qml.Wires): The device wires this circuit will run on.
        """
        n_qubits = len(wires)
        n_rotations = len(params)

        state = np.repeat([0], n_qubits)
        state[0:state_n] = 1
        qml.BasisState(state, wires=wires)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            if n_extra_rots > 0:
                # There may be "extra" parameter sets required for which it's not necessarily
                # to perform another full alternating cycle. Apply these to the qubits as needed.
                extra_params = params[-n_extra_rots:, :]
                extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
                qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

    num_qubits = len(H.wires)

    dev = qml.device('default.qubit', wires=num_qubits)

    from functools import partial
    cost_fns = [qml.ExpvalCost(partial(variational_ansatz, state_n=i), H, dev)
                for i in [0, 1, 2]]
    weights = np.array([5, 2, 1])

    def cost_fn(params):
        return [cost_fn(params) for cost_fn in cost_fns] @ weights

    opt = qml.AdamOptimizer(stepsize=0.4)
    # print('opt = qml.AdamOptimizer(stepsize=0.4)')

    max_iterations = 300
    rel_conv_tol = 1e-6

    num_param_sets = num_qubits * 3
    np.random.seed(42)
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2,
                               size=(num_param_sets, 3))

    #import time; clock = time.time()

    for n in range(max_iterations):
        params, prev_cost = opt.step_and_cost(cost_fn, params)
        cost = cost_fn(params)
        conv = np.abs((cost - prev_cost) / cost)

        # # DEBUG PRINT
        # if n % 20 == 0:
        #     energies = [cf(params) for cf in cost_fns]
        #     print(f'Iteration = {n}, cost = {cost} energies = ', energies,
        #           f'time {time.time() - clock:.0f}s')

        if conv <= rel_conv_tol:
            break

        energies = sorted([cf(params) for cf in cost_fns])

    # from scipy.optimize import minimize
    # cf_for_scipy = lambda params: cost_fn(np.reshape(params, (-1, 3)))
    # res = minimize(cf_for_scipy, np.ravel(params), method='COBYLA',
    #                options=dict(maxiter=1000))
    # print(res)
    # print('energies: ', [cf(np.reshape(res['x'], (-1, 3))) for cf in cost_fns])

    # print(f'tot time: ', time.time() - clock)

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
