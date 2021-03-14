from qiskit import IBMQ
from qiskit import execute, QuantumCircuit
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.analysis import hellinger_fidelity

from metrics import hog, l1_norm
from pass_manager import noise_pass_manager

qasm_file = 'path/to/qasm/file'

qc = QuantumCircuit.from_qasm_file(qasm_file)

provider = IBMQ.load_account()

backend_name = 'YOUR BACKEND NAME'

backend = provider.get_backend(backend_name)

properties = backend.properties()
coupling_map = backend.configuration().coupling_map
noise_model = NoiseModel.from_backend(backend)

sim_backend = QasmSimulator(method='statevector', max_parallel_shots=0, max_parallel_threads=0, noise_model=noise_model)

pass_manager = noise_pass_manager(coupling_map=coupling_map, layout_method='noise_adaptive',
                                  seed_transpiler=1000,
                                  routing_method='noise_adaptive', backend_properties=properties,
                                  alpha=0.5)

ideal_result = execute(qc, backend=StatevectorSimulator()).result()
ideal_counts = Statevector(ideal_result.get_statevector(qc)).sample_counts(8192)
ideal_probs = Statevector(ideal_result.get_statevector()).probabilities_dict()

qiskit_na_t_result = execute(qc, backend=sim_backend, shots=8192, pass_manager=pass_manager,
                             noise_model=noise_model).result()

sampled_counts = qiskit_na_t_result.get_counts(qc)

hellinger_fidelity = hellinger_fidelity(ideal_counts, sampled_counts)
hog = hog(sampled_counts, ideal_probs)
l1_norm = l1_norm(sampled_counts, ideal_probs)
