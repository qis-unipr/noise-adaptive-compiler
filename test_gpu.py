import os
import sys
import pickle as pkl

from qiskit import execute, QuantumCircuit, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Statevector
from qiskit.test.mock import FakeMelbourne, FakeBogota

from pass_manager import noise_pass_manager
from qiskit.quantum_info.analysis import hellinger_fidelity

from metrics import hog, cross_entropy, l1_norm

backend = FakeMelbourne()
properties = backend.properties()
coupling_map = backend.configuration().coupling_map
backend_name = backend.name()

backend_options = dict()
backend_options["method"] = "density_matrix_gpu"
backend_options["max_parallel_shots"] = 1
backend_options["max_parallel_threads"] = 1

transform = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'true':
        transform = True

layout_method = 'dense'
if len(sys.argv) > 2:
    layout_method = sys.argv[2]

routing_method = 'stochastic'
if len(sys.argv) > 3:
    routing_method = sys.argv[3]

metric = 'hellinger'
if len(sys.argv) > 4:
    metric = sys.argv[4]

if os.path.isfile('{}_results_{}_{}_{}.pkl'.format(backend_name, transform, layout_method, routing_method)):
    with open('{}_results_{}_{}_{}.pkl'.format(backend_name, transform, layout_method, routing_method), 'rb') as f:
        results = pkl.load(f)
else:
    results = dict()

for circuit in os.listdir('circuits'):
    if circuit.replace('.qasm', '') in results:
        continue
    qc = QuantumCircuit.from_qasm_file('circuits/{}'.format(circuit))

    if qc.num_qubits > backend.configuration().n_qubits:
        continue

    print(circuit)

    ideal_result = execute(qc, backend=Aer.get_backend('statevector_simulator')).result()
    ideal_counts = Statevector(ideal_result.get_statevector()).sample_counts(1024)
    ideal_probs = Statevector(ideal_result.get_statevector()).probabilities_dict()

    qc.measure_active()

    pass_manager = noise_pass_manager(backend=backend, layout_method=layout_method, routing_method='noise_adaptive',
                                      seed_transpiler=1000, transform=transform)
    compiled = pass_manager.run(qc)

    noise_result = execute(compiled, backend=QasmSimulator(), backend_properties=backend.properties(),
                           backend_options=backend_options, coupling_map=coupling_map, shots=1024, optimization_level=0,
                           noise_model=NoiseModel.from_backend(backend)).result()

    counts = noise_result.get_counts()
    fidelity = hellinger_fidelity(ideal_counts, counts)
    hog = hog(counts, ideal_probs)
    ce = cross_entropy(counts, ideal_probs)
    l1 = l1_norm(counts, ideal_probs)

    pass_manager = noise_pass_manager(backend=backend, layout_method=layout_method,
                                      seed_transpiler=1000, routing_method=routing_method)
    compiled = pass_manager.run(qc)

    qiskit_results = execute(compiled, backend=QasmSimulator(), backend_properties=backend.properties(),
                             backend_options=backend_options, coupling_map=coupling_map, shots=1024,
                             optimization_level=0, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                             noise_model=NoiseModel.from_backend(backend)).result()

    qiskit_counts = qiskit_results.get_counts()
    qiskit_fidelity = hellinger_fidelity(ideal_counts, counts)
    qiskit_hog = hog(counts, ideal_probs)
    qiskit_ce = cross_entropy(counts, ideal_probs)
    qiskit_l1 = l1_norm(counts, ideal_probs)

    results[circuit.replace('.qasm', '')] = {'noise': {'hellinger': fidelity, 'hog': hog, 'ce': ce, 'l1': l1},
                                             'qiskit': {'hellinger': qiskit_fidelity, 'hog': qiskit_hog, 'ce': qiskit_ce, 'l1': qiskit_l1}}

    with open('{}_results_{}_{}_{}.pkl'.format(backend_name, transform, layout_method, routing_method), 'wb') as f:
        pkl.dump(results, f)
