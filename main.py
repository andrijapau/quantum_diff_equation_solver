from tqdm import tqdm

from src.analytical_soln import *
from src.quantum_circuit import *

if __name__ == '__main__':
    t = np.linspace(0.01, 1.51, 10)
    # for t_ in t:
    num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits = 1, 4, 1

    p0, b = (1 / (np.sqrt(2))) * np.array([1, 1]), (1 / (np.sqrt(2))) * np.array([1, 1])

    x_C, y_C = [], []
    x_Q, y_Q = [], []
    for i in tqdm(range(len(t))):
        circuit = LDESolver(num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits, t[i])
        x_analytic, y_analytic = solve_p(t[i], p0, b)
        print("Analytic:", x_analytic, y_analytic)
        print("Predicted:", circuit.x, circuit.y)
        x_C.append(x_analytic)
        y_C.append(y_analytic)

        x_Q.append(circuit.x)
        y_Q.append(circuit.y)

    circuit.draw_circuit()

    plt.plot(t, x_C, 'r--', label='x(t) Analytic')
    plt.plot(t, x_Q, 'r', label='x(t) Quantum')

    plt.plot(t, y_C, 'b--', label='y(t) Analytic')
    plt.plot(t, y_Q, 'b', label='y(t) Quantum')

    plt.xlabel(r"$t$, time")
    plt.ylabel(r"$d$, distance")

    plt.legend(loc='best')
    plt.title("Position Components v.s. Time")

    plt.savefig("./results/position_vs_time.png", dpi=300)
    plt.show()

    plt.plot(x_C, y_C, 'k--', label='Analytic')
    plt.plot(x_Q, y_Q, 'r', label='Quantum')
    plt.legend(loc='best')
    plt.xlabel(r"$x(t)$")
    plt.ylabel(r"$y(t)$")
    plt.title("XY Phase Space")

    plt.savefig("./results/x_vs_y.png", dpi=300)
    plt.show()
