from tqdm import tqdm

from src.analytical_soln import *
from src.quantum_circuit import *

if __name__ == '__main__':
    t = np.linspace(0.01, 5.01, 10)
    # for t_ in t:
    num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits = 1, 3, 1

    p0, b = (1 / (np.sqrt(2))) * np.array([1, 1]), (1 / (np.sqrt(2))) * np.array([1, -1])

    x_C, y_C = [], []
    x_Q, y_Q = [], []
    for i in tqdm(range(len(t))):
        circuit = LDESolver(num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits, t[i])
        x_analytic, y_analytic = solve_p(t[i], p0, b)
        print(x_analytic, y_analytic)
        x_C.append(x_analytic)
        y_C.append(y_analytic)

        x_Q.append(circuit.x)
        y_Q.append(circuit.y)

    circuit.draw_circuit()

    plt.plot(t, x_C, 'k', label='Classical Solution')
    plt.plot(t, x_Q, 'r', label='Quantum Solution')
    plt.legend(loc='best')
    plt.show()

    plt.plot(t, y_C, 'k', label='Classical Solution')
    plt.plot(t, y_Q, 'r', label='Quantum Solution')
    plt.legend(loc='best')
    plt.show()
