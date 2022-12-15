from src.analytical_soln import *
from src.quantum_circuit import *

if __name__ == '__main__':

    # Define t space over which to solve DE
    t_arr = np.linspace(0.01, 2.51, 25)

    # Define circuit parameters
    num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits = 1, 2, 1

    # Define analytical solution parameters
    p0, b = (1 / (np.sqrt(2))) * np.array([1, 1]), (1 / (np.sqrt(2))) * np.array([1, 1])
    M = np.array([[0, 1], [1, 0]])

    # Define arrays to store information
    x_C, y_C = [], []
    x_Q, y_Q = [], []

    for t in t_arr:
        # Create circuit
        circuit = LDESolver(num_of_anc_reg1, num_of_anc_reg2, num_of_work_qubits, t)

        # Solve analytic expression
        x_analytic, y_analytic = solve_p(t, M, p0, b)

        print(f"Analytic x({round(t, 3)}) = {round(x_analytic, 3)}")
        print(f"Analytic y({round(t, 3)}) = {round(y_analytic, 3)}")
        print(f"Quantum x({round(t, 3)}) = {round(circuit.x, 3)}")
        print(f"Quantum y({round(t, 3)}) = {round(circuit.y, 3)}")

        # Store results
        x_C.append(x_analytic)
        y_C.append(y_analytic)
        x_Q.append(circuit.x)
        y_Q.append(circuit.y)

    # Draw circuit for user
    circuit.draw_circuit()

    # Plot results
    plt.plot(t_arr, x_C, 'r--', label='x(t) Analytic')
    plt.plot(t_arr, x_Q, 'r', label='x(t) Quantum')

    plt.plot(t_arr, y_C, 'b--', label='y(t) Analytic')
    plt.plot(t_arr, y_Q, 'b', label='y(t) Quantum')

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

    x_C = np.array(x_C)
    x_Q = np.array(x_Q)
    plt.plot(t_arr, np.abs((x_C - x_Q) / x_Q) * 100, 'k', label='Relative Error')
    plt.legend(loc='best')
    plt.xlabel(r"$t$, time")
    plt.ylabel(r"$|\Delta x / x|$, (%)")
    plt.title("Relative Error v.s. Time")
    plt.savefig("./results/error.png", dpi=300)
    plt.show()
