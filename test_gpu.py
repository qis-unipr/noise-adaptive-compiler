import os
import pickle as pkl

from qiskit import execute, QuantumCircuit, Aer
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Statevector
from qiskit.test.mock import FakeMelbourne, FakeBogota
from qiskit.transpiler import PassManager

from pass_manager import noise_pass_manager
from qiskit.quantum_info.analysis import hellinger_fidelity

backend = FakeMelbourne()
properties = backend.properties()
coupling_map = backend.configuration().coupling_map
backend_name = backend.name()

backend_options = dict()
backend_options["method"] = "density_matrix_gpu"
backend_options["max_parallel_shots"] = 1
backend_options["max_parallel_threads"] = 1

pass_manager = noise_pass_manager(backend=backend, layout_method='chain', routing_method='noise_adaptive',
                                  seed_transpiler=1000)

if os.path.isfile('{}_hellinger_results.pkl'.format(backend_name)):
    with open('{}_hellinger_results.pkl'.format(backend_name), 'rb') as f:
        results = pkl.load(f)
else:
    results = dict()

for circuit in os.listdir('circuits'):
    if circuit.replace('.qasm', '') in results:
        continue
    qc = QuantumCircuit.from_qasm_file('circuits/{}'.format(circuit))

    ideal_result = execute(qc, backend=Aer.get_backend('statevector_simulator')).result()
    ideal_counts = Statevector(ideal_result.get_statevector()).probabilities_dict()

    qc.measure_active()
    compiled = pass_manager.run(qc)

    noisy_result = execute(compiled, backend=QasmSimulator(), backend_properties=backend.properties(),
                           backend_options=backend_options, coupling_map=coupling_map, shots=1024,
                           pass_manager=PassManager()).result()

    fidelity = hellinger_fidelity(ideal_counts, noisy_result.get_counts())

    results[circuit.replace('.qasm', '')] = fidelity

    with open('{}_hellinger_results.pkl'.format(backend_name), 'wb') as f:
        pkl.dump(results, f)
