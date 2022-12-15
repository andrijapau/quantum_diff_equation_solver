import math

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
import numpy as np
from qiskit import Aer, IBMQ, QuantumCircuit, QuantumRegister, execute, transpile
from qiskit.circuit.library.standard_gates import XGate
from qiskit.quantum_info.operators import Operator


class LDESolver:
    '''
    Class that is responsible for the creation of the LDE Solver Circuit based on the paper,

    T. Xin et al., “Quantum algorithm for solving linear differential equations: Theory and experiment,”
    Physical Review A, vol. 101, no. 3, Mar. 2020, doi: 10.1103/physreva.101.032307.
    '''

    def __init__(self, num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits, t):

        self.shots = 10000

        self.load_account()

        self.create_circuit(num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits)

        # Creates the V and W unitaries
        self.create_VS_WS(t)
        self.create_V()

        # Encoding Section
        self.encode()

        # Entanglement Section
        self.entanglement()

        # Decoding Section
        self.decode()

        self.measure()

        self.execute()

        self.display_results()

    def execute(self):
        '''
        Executes the created quantum circuit on the IBMQ Qasm Simulator
        '''
        self.num_of_qubits = self.circuit.num_qubits

        self.backend = Aer.get_backend('qasm_simulator')
        job = execute(
            transpile(self.circuit, backend=self.backend),
            backend=self.backend,
            shots=self.shots
        )

        self.result = job.result()

    def draw_circuit(self):
        '''
        Draws the quantum circuit and saves it to the results folder.
        '''

        self.circuit.draw('mpl', style={'name': 'bw', 'dpi': 350}, filename="./results/circuit.png")
        plt.show()

    def display_results(self):
        '''
        Computes results as a class variable to be processed in main.
        '''

        counts = self.result.get_counts()
        probs = {string: count / self.shots for string, count in counts.items()}

        y_adj, x_adj = 0, 0
        if '0001' in probs.keys():
            y_adj = probs['0001']
        if '1001' in probs.keys():
            x_adj = probs['1001']

        self.y = (self.N ** 2) * np.sqrt(probs['1000'] + y_adj)
        self.x = (self.N ** 2) * np.sqrt(probs['0000'] + x_adj)

    def measure(self):
        '''
        Measures the quantum circuit over all qubits.
        '''

        self.circuit.measure_all()

    def entanglement(self):
        '''
        Calculates the entanglement operator for a given order.
        '''

        order = 3
        num_of_ancilla = int(np.log2(order + 1))
        for i in range(2 ** num_of_ancilla):
            if i % 2 != 0:
                ctrl = "".join(bin(i)[2:][::-1].ljust(num_of_ancilla, '0'))
                self.circuit.append(
                    XGate().control(num_ctrl_qubits=num_of_ancilla, ctrl_state=ctrl),
                    [self.anc_reg2[0], self.anc_reg2[1], self.work_qubit_reg[-1]]
                )

    def encode(self):
        '''
        Encoding part of the quantum circuit.
        '''

        # Create a seperate circuit to turn into VS1 gate
        S1_circuit = QuantumCircuit(3)

        # Ux
        S1_circuit.h(2)

        # Vs1
        S1_circuit.unitary(self.v_S1_U, [0, 1], label='VS1')
        S1_gate = S1_circuit.to_gate().control(1, ctrl_state='0')

        # Create a seperate circuit to turn into VS1 gate
        S2_circuit = QuantumCircuit(3)

        # Ub
        S2_circuit.h(2)

        # Vs2
        S2_circuit.unitary(self.v_S2_U, [0, 1], label='VS2')
        S2_gate = S2_circuit.to_gate().control(1, ctrl_state='1')

        # Attach encoding part of circuit.
        self.circuit.unitary(self.V_U, 0, label='V')
        self.circuit.append(S1_gate, [0, 1, 2, 3])
        self.circuit.append(S2_gate, [0, 1, 2, 3])

    def decode(self):
        '''
        Decoding part of the quantum circuit.
        '''

        # WS1
        S1_circuit = QuantumCircuit(3)
        S1_circuit.unitary(self.w_S1_U, [0, 1], label='WS1')
        S1_gate = S1_circuit.to_gate().control(1, ctrl_state='0')

        # WS2
        S2_circuit = QuantumCircuit(3)
        S2_circuit.unitary(self.w_S2_U, [0, 1], label='WS2')
        S2_gate = S2_circuit.to_gate().control(1, ctrl_state='1')

        # Append decoding part onto circuit.
        self.circuit.append(S1_gate, [0, 1, 2, 3])
        self.circuit.append(S2_gate, [0, 1, 2, 3])
        self.circuit.unitary(self.W_U, 0, label='W')

    def create_V(self):
        '''
        Creates the V and W operators
        '''

        self.N = np.sqrt(self.C ** 2 + self.D ** 2)
        V = (1 / self.N) * np.array([[self.C, self.D], [self.D, -self.C]])

        self.V_U = Operator(V)
        self.W_U = self.V_U.transpose()

    def create_VS_WS(self, t):
        '''
        Creates the V and W matrices using Gram-Schmidt method.
        '''

        v_S1 = np.array(
            [[np.sqrt(self.e(t, 0)),
              np.sqrt(self.e(t, 1)),
              np.sqrt(self.e(t, 2)),
              np.sqrt(self.e(t, 3))]]
        )

        v_S2 = np.array(
            [[np.sqrt(self.e(t, 1)),
              np.sqrt(self.e(t, 2)),
              np.sqrt(self.e(t, 3)),
              0]]
        )

        self.C = np.sqrt(np.sum(v_S1 ** 2, axis=1)[0])
        self.D = np.sqrt(np.sum(v_S2 ** 2, axis=1)[0])

        v_S1 = (1 / self.C) * v_S1
        v_S2 = (1 / self.D) * v_S2

        self.v_S1_U = Operator(self.create_unitary(v_S1))
        self.v_S2_U = Operator(self.create_unitary(v_S2))

        self.w_S1_U, self.w_S2_U = self.v_S1_U.transpose(), self.v_S2_U.transpose()

    def e(self, t, index):
        '''
        Term in the Taylor Expansion of exp(x) used in W and V matrices.
        '''
        return (t ** index) / (math.factorial(index))

    def create_unitary(self, v):
        '''
        Function that calculates the unitary matrix for a given vector in the first column. This function is from,
        https://math.stackexchange.com/questions/4160055/create-a-unitary-matrix-out-of-a-column-vector
        '''

        dim = v.size
        # Return identity if v is a multiple of e1
        if v[0][0] and not np.any(v[0][1:]):
            return np.identity(dim)
        e1 = np.zeros(dim)
        e1[0] = 1
        w = v / np.linalg.norm(v) - e1
        return np.identity(dim) - 2 * ((np.dot(w.T, w)) / (np.dot(w, w.T)))

    def load_account(self):
        '''
        Loads my IBMQ account.
        '''
        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='strangeworks-hub', group="science-team", project="science-test")

    def create_circuit(self, num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits):
        '''
        Define and set-up quantum circuit.
        '''

        self.circuit = QuantumCircuit(name='LDE_Solver_Circuit')

        self.anc_reg1 = QuantumRegister(num_of_anc_reg1, name='anc1')

        self.anc_reg2 = QuantumRegister(num_of_anc_reg2, name='anc2')

        self.work_qubit_reg = QuantumRegister(num_of_work_qubits, name='work')

        self.circuit.add_register(self.anc_reg1, self.anc_reg2, self.work_qubit_reg)
