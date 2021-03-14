from qiskit.converters import circuit_to_dag
from qiskit import transpile


def cx_depth(circuit):
    """

    Args:
        op (qiskit.circuit.Instruction):
        circuit (qiskit.circuit.QuantumCircuit):

    Returns:

    """
    circuit = transpile(circuit, basis_gates=['u3', 'cx'])
    dag = circuit_to_dag(circuit)
    gates = [gate for gate in dag.nodes() if gate.type == 'op']
    for gate in gates:
        if gate.op.name != 'cx':
            dag.remove_op_node(gate)
    return dag.depth()


def cx_count(circuit):
    circuit = transpile(circuit, basis_gates=['u3', 'cx'])
    return len(circuit_to_dag(circuit).two_qubit_ops())
