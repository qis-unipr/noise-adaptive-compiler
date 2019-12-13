# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from qiskit import execute, Aer, QuantumCircuit
from qiskit.extensions import SwapGate
from qiskit.providers.aer.noise.device import basic_device_noise_model
from qiskit.test.base import dicts_almost_equal
from qiskit.test.mock import FakeMelbourne  # NB will need to install dev requirements
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout, \
    BarrierBeforeFinalMeasurements, Unroll3qOrMore, Decompose, CheckCXDirection, CXDirection, Depth, FixedPoint, \
    RemoveResetInZeroState, ConsolidateBlocks, Collect2qBlocks, Unroller, Optimize1qGates, CommutativeCancellation, \
    OptimizeSwapBeforeMeasure, RemoveDiagonalGatesBeforeMeasure

from passes import ChainLayout, TransformCxCascade, NoiseAdaptiveSwap

"""
Implementation based on qiskit 0.13.0

The proposed solution comprises three passes, one analysis pass
to find an initial mapping and two transformation passes.

The analysis pass is called ChainLayout. Its purpose is to find an initial
mapping where the i-th qubit has a connection with the (i-1)-th and (i+1)-th
qubits. Due to connectivity issues, such a layout is not possible when the
circuit uses all the available qubits in the device. In such cases this pass
will add unavoidable swaps.

The first transformation pass is called TransformCxCascade.
This pass is able to identify CNOT cascades and transform them
into nearest-neighbor CNOT sequences, which can then be easily mapped onto a
ChainLayout. If the circuit is characterized by many consecutive CNOT cascades,
there is the chance for significant circuit optimization after the
transformation is done. This pass must be run before applying any layout and
requires the Unroll pass to be run before. It is also able to identify what
I call inverse CNOT cascades (more information are available in the docstrings).

The last transformation pass is the actual swapper, which is denoted as
NoiseAdaptiveSwap. The pass starts by computing the swap path and reliability
of such path between every pair of qubits.
Then the pass cycles through the circuit layers, and every time it encounters
a remote CNOT, it runs the actual algorithm. It computes n possible swaps
interesting only the qubits involved by the remote CNOT, where n is 4 by default
but can be set by the user. These swaps are ranked 1) by how many of
the m following CNOT can be executed, where m is 10 by default but can be
also set by the user, and 2) by the reliability of the complete precomputed
swap path (as an estimation of the path that will be chosen).
For every one of such possible swaps, the algorithm is repeated until a depth
search of d is reached, where d is 4 by default but can be set by the user.
At the end of the search, the search path with highest swap score is used
to change the layout and continue with the algorithm until no more remote CNOTs
are found.
"""

"""
Your solution needs to comprise of one or more passes that you have written along with
a PassManager that uses them. The PassManager is allowed to use passes that are already
included in Qiskit. 
"""

""" To test your passes you can use the fake backend classes. Your solutions will be tested against
Yorktown, Ourense and Melbourne, as well as some internal backends. """
backend = FakeMelbourne()
properties = backend.properties()
coupling_map = backend.configuration().coupling_map

""" You must submit a pass manager which uses at least one pass you have written. 
Examples of creating more complex pass managers can be seen in qiskit.transpiler.preset_passmanagers"""

"""
This pass manager is basically the level 3 preset pass manager. The DenseLayout
has been substituted by the TransformCxCascade and ChainLayout passes.
The StochasticSwap has been changed to the NoiseAdaptiveSwap.
"""
pass_manager = PassManager()
pass_manager.append([
    TransformCxCascade(),
    ChainLayout(coupling_map=coupling_map, backend_prop=properties),
    FullAncillaAllocation(coupling_map), EnlargeWithAncilla(),
    ApplyLayout()
])


def _direction_condition(property_set):
    return not property_set['is_direction_mapped']

pass_manager.append([
    BarrierBeforeFinalMeasurements(),
    Unroll3qOrMore(),
    NoiseAdaptiveSwap(coupling_map=coupling_map, backend_prop=properties),
    Decompose(SwapGate)
])


def direction_condition(property_set):
    return not property_set['is_direction_mapped']

if not coupling_map.is_symmetric:
    pass_manager.append(CheckCXDirection(coupling_map))
    pass_manager.append(CXDirection(coupling_map), condition=direction_condition)

depth_check = [Depth(), FixedPoint('depth')]


def opt_control(property_set):
    return not property_set['depth_fixed_point']

basis_gates = ['u1', 'u2', 'u3', 'cx']
opt = [
    RemoveResetInZeroState(),
    Collect2qBlocks(), ConsolidateBlocks(),
    Unroller(basis_gates),
    Optimize1qGates(), CommutativeCancellation(),
    OptimizeSwapBeforeMeasure(), RemoveDiagonalGatesBeforeMeasure()
]

if coupling_map and not coupling_map.is_symmetric:
    opt.append(CXDirection(coupling_map))
pass_manager.append(depth_check + opt, do_while=opt_control)

""" This allows us to simulate the noise a real device has, so that you don't have to wait for jobs to complete
on the actual backends."""
noise_model = basic_device_noise_model(properties)
simulator = Aer.get_backend('qasm_simulator')

""" This is the circuit we are going to look at"""
qc = QuantumCircuit(2, 2)
qc.h(1)
qc.measure(0, 0)
circuits = [qc]

result_normal = execute(circuits,
                        simulator,
                        coupling_map=coupling_map).result().get_counts()

# NB we include the noise model in the second run
result_noisy = execute(circuits,
                       simulator,
                       noise_model=noise_model,
                       coupling_map=coupling_map).result().get_counts()

""" Check to see how similar the counts from the two runs are, where delta the allowed difference between
the counts. Returns an empty string if they are almost equal, otherwise returns an error message which can 
then be printed."""
equality_check = dicts_almost_equal(result_normal, result_noisy, delta=1e-8)

if equality_check:
    print(equality_check)
