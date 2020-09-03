from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT


def bv(n):
    s = ''
    bit = '0'
    for i in range(n):
        if bit == '0':
            bit = '1'
        else:
            bit = '0'
        s += bit

    bv_circuit = QuantumCircuit(n+1)

    # put ancilla in state |->
    bv_circuit.h(n)
    bv_circuit.z(n)

    # Apply Hadamard gates before querying the oracle
    for i in range(n):
        bv_circuit.h(i)

    # Apply barrier
    bv_circuit.barrier()

    # Apply the inner-product oracle
    s = s[::-1]  # reverse s to fit qiskit's qubit ordering
    for q in range(n):
        if s[q] == '0':
            bv_circuit.i(q)
        else:
            bv_circuit.cx(q, n)

    # Apply barrier
    bv_circuit.barrier()

    # Apply Hadamard gates after querying the oracle
    for i in range(n):
        bv_circuit.h(i)

    return bv_circuit


n_qubits = [2, 3, 4, 5, 6, 8, 12, 16, 20]

for n in n_qubits:

    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(0, n-1):
        qc.cx(0, q+1)
    with open('circuits/ghz_{}.qasm'.format(n), 'w') as f:
        f.write(qc.qasm())

    qc = QFT(n)
    with open('circuits/qft_{}.qasm'.format(n), 'w') as f:
        f.write(qc.qasm())

    qc = bv(n)
    with open('circuits/bv_{}.qasm'.format(n), 'w') as f:
        f.write(qc.qasm())


