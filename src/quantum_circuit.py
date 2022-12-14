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
    def __init__(self, num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits, t):
        self.shots = 10000

        self.load_account()

        self.create_circuit(num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits)

        self.create_VS_WS(t)
        self.create_V()

        self.encode()

        self.entanglement()

        self.decode()

        self.measure()

        self.execute()

        # self.draw_circuit()
        self.display_results()

    def execute(self):
        self.num_of_qubits = self.circuit.num_qubits
        # self.backend = least_busy(
        #     self.provider.backends(filters=lambda x: x.configuration().n_qubits >= self.num_of_qubits
        #                                              and not x.configuration().simulator
        #                                              and x.status().operational == True)
        # )
        self.backend = Aer.get_backend('qasm_simulator')
        job = execute(
            transpile(self.circuit, backend=self.backend),
            backend=self.backend,
            shots=self.shots
        )

        self.result = job.result()

    def draw_circuit(self):
        """"""
        self.circuit.draw('mpl', style={'name': 'bw', 'dpi': 350}, filename="./results/circuit.png")
        plt.show()

    def display_results(self):
        counts = self.result.get_counts()
        # print(self.N2)
        probs = {string: count / self.shots for string, count in counts.items()}
        print(probs, self.N)
        # self.x = min(probs['100000'], probs['000001'])
        y_adj, x_adj = 0, 0
        if '0001' in probs.keys():
            y_adj = probs['0001']
        if '1001' in probs.keys():
            x_adj = probs['1001']
        self.y = (self.N ** 2) * np.sqrt(probs['1000'] + y_adj)
        self.x = (self.N ** 2) * np.sqrt(probs['0000'] + x_adj)
        # self.x = probs['00001']
        # self.y = min(probs['000000'], probs['100001'])
        # plot_histogram(self.result.get_counts(), title="LDE Solver", color='black')
        # plt.show()

    def measure(self):
        # self.circuit.measure(self.work_qubit_reg, self.classical_reg)
        self.circuit.measure_all()

    def entanglement(self):
        order = 2
        for i in range(2 ** order):
            # print(self.circuit)
            if i % 2 != 0:
                ctrl = "".join(bin(i)[2:][::-1].ljust(order, '0'))
                self.circuit.append(
                    XGate().control(num_ctrl_qubits=order, ctrl_state=ctrl),
                    [self.anc_reg2[0], self.anc_reg2[1], self.work_qubit_reg[-1]]
                )
        # self.circuit.append(
        #     XGate().control(num_ctrl_qubits=3, ctrl_state=3),
        #     [self.anc_reg2[0], self.anc_reg2[1], self.anc_reg2[2], self.work_qubit_reg[-1]]
        # )
        # self.circuit.append(
        #     XGate().control(num_ctrl_qubits=3, ctrl_state=5),
        #     [self.anc_reg2[0], self.anc_reg2[1], self.anc_reg2[2], self.work_qubit_reg[-1]]
        # )
        # self.circuit.append(
        #     XGate().control(num_ctrl_qubits=3, ctrl_state=7),
        #     [self.anc_reg2[0], self.anc_reg2[1], self.anc_reg2[2], self.work_qubit_reg[-1]]
        # )

    # self.circuit.cx(self.anc_reg2, self.work_qubit_reg, ctrl_state='1')
    # self.circuit.cx(self.anc_reg2, self.work_qubit_reg, ctrl_state='1')

    def encode(self):
        S1_circuit = QuantumCircuit(3)
        # Ux Ub
        S1_circuit.h(2)
        # Vs1 Vs2
        S1_circuit.unitary(self.v_S1_U, [0, 1], label='VS1')
        S1_gate = S1_circuit.to_gate().control(1, ctrl_state='0')

        S2_circuit = QuantumCircuit(3)
        # S2_circuit.x(4)
        S2_circuit.h(2)
        S2_circuit.unitary(self.v_S2_U, [0, 1], label='VS2')
        S2_gate = S2_circuit.to_gate().control(1, ctrl_state='1')
        # print(S2_circuit)

        self.circuit.unitary(self.V_U, 0, label='V')
        self.circuit.append(S1_gate, [0, 1, 2, 3])
        self.circuit.append(S2_gate, [0, 1, 2, 3])

    def decode(self):

        S1_circuit = QuantumCircuit(3)
        # Vs1 Vs2
        S1_circuit.unitary(self.w_S1_U, [0, 1], label='WS1')
        S1_gate = S1_circuit.to_gate().control(1, ctrl_state='0')

        # print(S1_circuit)
        S2_circuit = QuantumCircuit(3)
        S2_circuit.unitary(self.w_S2_U, [0, 1], label='WS2')
        S2_gate = S2_circuit.to_gate().control(1, ctrl_state='1')
        # print(S2_circuit)
        self.circuit.append(S1_gate, [0, 1, 2, 3])
        self.circuit.append(S2_gate, [0, 1, 2, 3])
        self.circuit.unitary(self.W_U, 0, label='W')

        # S1_circuit = QuantumCircuit(3)
        # # Ws1
        # S1_circuit.unitary(self.w_S1_U, [0, 1, 2], label='WS1')
        # S1_gate = S1_circuit.to_gate().control(1, ctrl_state='0')
        # # Ws2
        # S2_circuit = QuantumCircuit(3)
        # S2_circuit.unitary(self.w_S2_U, [0, 1, 2], label='WS2')
        # S2_gate = S2_circuit.to_gate().control(1, ctrl_state='1')
        #
        # self.circuit.append(S1_gate, [0, 1, 2, 3])
        # self.circuit.append(S2_gate, [0, 1, 2, 3])
        # self.circuit.unitary(self.W_U, 0, label='W')

    def create_V(self):
        self.N = np.sqrt(self.C ** 2 + self.D ** 2)
        V = (1 / self.N) * np.array([[self.C, self.D], [self.D, -self.C]])

        self.V_U = Operator(V)
        self.W_U = self.V_U.transpose()

    def create_VS_WS(self, t):

        order = 0  # Doesn't matter

        v_S1 = np.array(
            [[np.sqrt(self.e(t, 0, order, noSum=True, index=0)),
              np.sqrt(self.e(t, 1, order, noSum=True, index=1)),
              np.sqrt(self.e(t, 2, order, noSum=True, index=2)),
              np.sqrt(self.e(t, 3, order, noSum=True, index=3))]]
        )

        v_S2 = np.array(
            [[np.sqrt(self.e(t, 1, order, noSum=True, index=1)),
              np.sqrt(self.e(t, 2, order, noSum=True, index=2)),
              np.sqrt(self.e(t, 3, order, noSum=True, index=3)),
              0]]
        )

        self.C = np.sqrt(np.sum(v_S1 ** 2, axis=1)[0])
        self.D = np.sqrt(np.sum(v_S2 ** 2, axis=1)[0])

        v_S1 = (1 / self.C) * v_S1
        v_S2 = (1 / self.D) * v_S2

        self.v_S1_U = Operator(self.create_unitary(v_S1))
        self.v_S2_U = Operator(self.create_unitary(v_S2))

        self.w_S1_U, self.w_S2_U = self.v_S1_U.transpose(), self.v_S2_U.transpose()

    def e(self, t, start, order, noSum=False, index=None):
        if not noSum:
            sum = 0
            for i in range(start, order):
                sum += (t ** i) / (math.factorial(i))

            return sum
        elif noSum:
            return (t ** index) / (math.factorial(index))

    def create_unitary(self, v):
        dim = v.size
        # Return identity if v is a multiple of e1
        if v[0][0] and not np.any(v[0][1:]):
            return np.identity(dim)
        e1 = np.zeros(dim)
        e1[0] = 1
        w = v / np.linalg.norm(v) - e1
        return np.identity(dim) - 2 * ((np.dot(w.T, w)) / (np.dot(w, w.T)))

    def load_account(self):
        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='strangeworks-hub', group="science-team", project="science-test")

    def create_circuit(self, num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits):
        self.circuit = QuantumCircuit(name='LDE_Solver_Circuit')

        self.anc_reg1 = QuantumRegister(num_of_anc_reg1, name='anc1')

        self.anc_reg2 = QuantumRegister(num_of_anc_reg2, name='anc2')

        self.work_qubit_reg = QuantumRegister(num_of_work_qubits, name='work')

        # self.classical_reg = ClassicalRegister(num_of_work_qubits, name='result')

        # self.circuit.add_register(self.anc_reg1, self.anc_reg2, self.work_qubit_reg, self.classical_reg)
        self.circuit.add_register(self.anc_reg1, self.anc_reg2, self.work_qubit_reg)
