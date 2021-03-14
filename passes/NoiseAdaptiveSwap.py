import logging
import math
from copy import deepcopy
from functools import reduce

import networkx as nx

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.extensions import SwapGate
from qiskit.transpiler import TransformationPass, TranspilerError, Layout, CouplingMap

logger = logging.getLogger(__name__)


class NoiseAdaptiveSwap(TransformationPass):

    def __init__(self, coupling_map, backend_prop, **kwargs):
        """
        NoiseAdaptiveSwap initializer.

        Args:
            coupling_map (CouplingMap or list): Directed graph or list representing a coupling map.
            backend_prop (qiskit.providers.models.BackendProperties):
        Keyword Args:
            search_depth (int): next layout search depth, default is 4.
            n_swaps (int): possible swaps considered in each layout, default is 4.
            next_gates (int): number of following gates used to score possible swaps, default is 10.
        Raises:
            TranspilerError: if invalid options.
        """
        super().__init__()

        if 'search_depth' in kwargs:
            self._search_depth = kwargs['search_depth']
        else:
            self._search_depth = 4
        if 'n_swaps' in kwargs:
            self._n_swaps = kwargs['n_swaps']
        else:
            self._n_swaps = 4
        if 'next_gates' in kwargs:
            self._next_gates = kwargs['next_gates']
        else:
            self._next_gates = 5
        if kwargs['alpha'] > 1 or kwargs['alpha'] < 0:
            raise TranspilerError('Must provide a valid alpha for Swap score mode != 0')
        self._alpha = kwargs['alpha']
        if 'readout' in kwargs:
            self._readout = kwargs['readout']
        else:
            self._readout = False
        if 'front' in kwargs:
            self._front = kwargs['front']
        else:
            self._front = False

        self._qreg = None
        if isinstance(coupling_map, list):
            self._coupling_map = CouplingMap(couplinglist=coupling_map)
        elif isinstance(coupling_map, CouplingMap):
            self._coupling_map = coupling_map
        else:
            raise TranspilerError('Coupling map of type %s is not a valid option.' % coupling_map.__class__)
        self._coupling_graph = self._coupling_map.graph.to_undirected()

        p, distances = nx.algorithms.shortest_paths.dense.floyd_warshall_predecessor_and_distance(self._coupling_graph)
        self._max_distance = 0
        for x in distances:
            for y in distances[x]:
                if distances[x][y] > self._max_distance:
                    self._max_distance = distances[x][y]
        p.clear()
        distances.clear()

        self.backend_prop = backend_prop
        self.swap_graph = nx.DiGraph()
        self.cx_reliability = {}
        self.swap_reliabs = {}

        if self._readout:
            self._readout_reliability = dict()
            i = 0
            for q in backend_prop.qubits:
                for info in q:
                    if info.name == 'readout_error':
                        self._readout_reliability[i] = 1.0 - info.value
                        i += 1
            #print(self._readout_reliability)

        backend_prop = self.backend_prop
        for ginfo in backend_prop.gates:
            if ginfo.gate == 'cx':
                for item in ginfo.parameters:
                    if item.name == 'gate_error':
                        g_reliab = 1.0 - item.value
                        break
                    else:
                        g_reliab = 1.0
                swap_reliab = pow(g_reliab, 3)
                # convert swap reliability to edge weight
                # for the Floyd-Warshall shortest weighted paths algorithm
                swap_cost = -math.log(swap_reliab) if swap_reliab != 0 else 10**(-10)
                self.swap_graph.add_edge(ginfo.qubits[0], ginfo.qubits[1], weight=swap_cost)
                self.swap_graph.add_edge(ginfo.qubits[1], ginfo.qubits[0], weight=swap_cost)
                self.cx_reliability[(ginfo.qubits[0], ginfo.qubits[1])] = g_reliab
                if self._readout:
                    qubits_readout_reliab = self._readout_reliability[ginfo.qubits[0]] * self._readout_reliability[
                        ginfo.qubits[1]]
                    self.cx_reliability[(ginfo.qubits[0], ginfo.qubits[1])] *= qubits_readout_reliab

        self.swap_paths, swap_reliabs_temp = nx.algorithms.shortest_paths.dense. \
            floyd_warshall_predecessor_and_distance(self.swap_graph, weight='weight')
        for i in swap_reliabs_temp:
            self.swap_reliabs[i] = {}
            for j in swap_reliabs_temp[i]:
                if (i, j) in self.cx_reliability:
                    self.swap_reliabs[i][j] = self.cx_reliability[(i, j)]
                elif (j, i) in self.cx_reliability:
                    self.swap_reliabs[i][j] = self.cx_reliability[(j, i)]
                else:
                    best_reliab = 0.0
                    for n in self.swap_graph.neighbors(j):
                        if (n, j) in self.cx_reliability:
                            reliab = math.exp(-swap_reliabs_temp[i][n]) * self.cx_reliability[(n, j)]
                        else:
                            reliab = math.exp(-swap_reliabs_temp[i][n]) * self.cx_reliability[(j, n)]
                        if reliab > best_reliab:
                            best_reliab = reliab
                    self.swap_reliabs[i][j] = best_reliab
        min_reliab = 1.0
        max_reliab = 0.0
        for i in self.swap_reliabs:
            for j in self.swap_reliabs[i]:
                if self.swap_reliabs[i][j] < min_reliab:
                    min_reliab = self.swap_reliabs[i][j]
                if self.swap_reliabs[i][j] > max_reliab:
                    max_reliab = self.swap_reliabs[i][j]

        for i in self.swap_reliabs:
            for j in self.swap_reliabs[i]:
                self.swap_reliabs[i][j] = (self.swap_reliabs[i][j]-min_reliab)/(max_reliab-min_reliab)
        logger.debug('Swap paths: %s' % str(self.swap_paths))

    def run(self, dag):
        """
        Run NoiseAdaptiveSwap on `dag`.
            Args:
                dag (DAGCircuit): the directed acyclic graph to be mapped.
            Returns:
                DAGCircuit: A mapped DAG.
            Raises:
                TranspilerError: if the coupling map or the layout are not
                compatible with the DAG.
        """
        # from qiskit.converters import dag_to_circuit
        # #print(dag_to_circuit(dag).qasm())
        new_dag = DAGCircuit()
        new_dag.name = dag.name

        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('Basic swap runs on physical circuits only')

        if len(dag.qubits()) > self._coupling_map.size():
            raise TranspilerError('The layout does not match the amount of qubits in the DAG')

        canonical_register = dag.qregs['q']
        self._qreg = deepcopy(canonical_register)
        new_dag.add_qreg(self._qreg)
        for c_reg in dag.cregs.values():
            new_dag.add_creg(deepcopy(c_reg))

        layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = layout.copy()

        gates = [[n for n in gate['graph'].nodes() if n.type == 'op'][0] for gate in dag.serial_layers()]

        executed = []
        to_execute = gates.copy()

        if self._front:
            to_execute, to_map, executed = self.update_to_map([], current_layout, to_execute)
            while to_map:
                #print('Searching Layout')
                next_step = self.search_layout(to_map, current_layout, to_execute, iter=self._search_depth)
                logger.info('Next step: %s' % str(next_step))
                current_layout = next_step['layout']
                to_map = next_step['to_map']
                to_execute = next_step['to_execute']
                executed.extend(next_step['executed'])
        else:
            while to_execute:
                next_step = self.search_layout([], current_layout, to_execute, iter=self._search_depth)
                logger.info('Next step: %s' % str(next_step))
                current_layout = next_step['layout']
                to_execute = next_step['to_execute']
                executed.extend(next_step['executed'])

        for gate in executed:
            new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs, gate.condition)

        return new_dag

    def update_to_map(self, to_map, layout, gates):
        busy = set()
        executed = []
        to_execute = []
        temp = to_map.copy()
        temp.extend(gates)
        to_map = []
        for gate in temp:
            qargs = gate.qargs

            if gate.name in ["barrier", "snapshot", "save", "load", "noise"]:
                if not qargs:
                    continue
                if busy.intersection(qargs):
                    busy.update(qargs)
                    to_execute.append(gate)
                else:
                    executed.append(self.execute_gate(gate, layout))
                continue

            if not busy.intersection(qargs):
                if len(qargs) == 1:
                    executed.append(self.execute_gate(gate, layout))
                elif self._coupling_map.distance(*[layout[q] for q in qargs]) == 1:
                    logger.debug('Executed two-qubit gate with qargs: %s\n' % gate.qargs)
                    executed.append(self.execute_gate(gate, layout))
                else:
                    logger.debug('Remote gate with qargs: %s\n' % gate.qargs)
                    to_map.append(gate)
                    busy.update(qargs)
            else:
                to_execute.append(gate)
                busy.update(qargs)

        return to_execute, to_map, executed

    def search_layout(self, to_map, layout, gates, iter=4, last_swap=None):
        """

        Args:
            layout (Layout): the current layout.
            gates (list): gates to be executed.
            iter (int): number of consecutive swaps to search before returning a solution.

        Returns:
            (dict): the solution found.
                to_execute (list): gates that could not be executed.
                executed (list): gates executed.
                score (tuple): score of the solution as (# of executed gates, reliability of inserted swaps).
                layout (Layout): layout of the solution.
        """
        if self._front:
            to_execute, to_map, executed = self.update_to_map(to_map, layout, gates)
        else:
            to_execute, to_map, executed = self.update_to_execute(gates, layout)

        current_step = {
            'to_map': to_map,
            'to_execute': to_execute,
            'executed': executed,
            'score': 1,
            'layout': layout,
            'swaps': last_swap
        }

        if iter == 0 or not to_map:
            return current_step

        if self._front:
            possible_swaps = self.new_possible_swaps(to_map, layout, to_execute, last_swap=last_swap)
        else:
            possible_swaps = self.possible_swaps(to_map[0], layout, to_execute, n=self._n_swaps)

        next_swap, best_step = None, None
        for swap in possible_swaps:
            current_layout = layout.copy()
            current_layout.swap(*swap['swap'])

            if self._front:
                next_step = self.search_layout(to_map, current_layout, to_execute, iter - 1, last_swap=swap['swap'])
            else:
                next_step = self.search_layout([], current_layout, to_execute, iter - 1, last_swap=swap['swap'])

            next_step['score'] = (swap['score']*next_step['score'])
            if next_swap is None or next_step['score'] > best_step['score']:
                next_swap, best_step = swap, next_step

        swap_gate = self.execute_swap_gate(next_swap)

        best_step = {
            'to_map': best_step['to_map'],
            'to_execute': best_step['to_execute'],
            'executed': executed + [swap_gate] + best_step['executed'],
            'score': best_step['score'],
            'layout': best_step['layout']
        }

        return best_step

    def execute_swap_gate(self, swap):
        """

        Args:
            swap (dict): swap to be executed.

        Returns:
            swap_gate (Gate): executed swap gate.
        """

        class Gate:
            def __init__(self, swap, qreg):
                self.op = SwapGate()
                self.qargs = [qreg[qarg] for qarg in swap['swap']]
                self.cargs = []
                self.condition = None

        return Gate(swap, self._qreg)

    def update_to_execute(self, gates, layout):
        """

        Args:
            gates (list): gates to be executed.
            layout (Layout): current circuit layout.

        Returns:
            (tuple):
                to_execute (list): gates to be executed.
                to_map (list): list of the front layer of gates that cannot be executed.
                executed (list): list of executed gates.
        """
        busy = set()
        executed = []
        to_execute = []
        to_map = []
        for gate in gates:
            qargs = gate.qargs

            if gate.name in ["barrier", "snapshot", "save", "load", "noise"]:
                if not qargs:
                    continue
                if busy.intersection(qargs):
                    busy.update(qargs)
                    to_execute.append(gate)
                else:
                    executed.append(self.execute_gate(gate, layout))
                continue

            if not busy.intersection(qargs):
                if len(qargs) == 1:
                    executed.append(self.execute_gate(gate, layout))
                elif self._coupling_map.distance(*[layout[q] for q in qargs]) == 1:
                    logger.debug('Executed two-qubit gate with qargs: %s\n' % gate.qargs)
                    executed.append(self.execute_gate(gate, layout))
                else:
                    logger.debug('Remote gate with qargs: %s\n' % gate.qargs)
                    to_execute.append(gate)
                    to_map.append(gate)
                    busy.update(qargs)
            else:
                to_execute.append(gate)
                busy.update(qargs)
        return to_execute, to_map, executed

    def new_possible_swaps(self, to_map, layout, to_execute, n=4, last_swap=None):
        """

        Args:
            remote_cnot (DAGNode): remote cnot.
            layout (Layout): current circuit layout.
            to_execute (list): gates to be executed.
            n (int): number of possible swaps to consider.

        Returns:
            possible_swaps (list): list of possible swaps ranked by their score.
        """
        temp_layout = layout.copy()
        qubits = []
        swap_qubits = []
        if last_swap:
            swap_qubits.append((last_swap[0], last_swap[1]))
        for gate in to_map:
            virt_qargs = gate.qargs
            qubits.extend([self.get_phys_qubit(q, temp_layout) for q in virt_qargs])
        possible_swaps = []

        for q in qubits:
            for v in self._coupling_graph[q]:
                if (q, v) in swap_qubits or (v, q) in swap_qubits:
                    continue
                swap = dict()
                swap['swap'], swap['score'] = self.score_swap([q, v], temp_layout, to_execute,
                                                              next_gates=self._next_gates, to_map=to_map)
                possible_swaps.append(swap)
                swap_qubits.append((q,v))
        possible_swaps = sorted(possible_swaps, key=lambda x: x['score'], reverse=True)

        return possible_swaps[:n]

    def possible_swaps(self, remote_cnot, layout, to_execute, n=4):
        """

        Args:
            remote_cnot (DAGNode): remote cnot.
            layout (Layout): current circuit layout.
            to_execute (list): gates to be executed.
            n (int): number of possible swaps to consider.

        Returns:
            possible_swaps (list): list of possible swaps ranked by their score.
        """
        if n < 1:
            raise TranspilerError('Invalid option for possible number of swaps.')

        temp_layout = layout.copy()
        virt_qargs = remote_cnot.qargs
        qubits = [self.get_phys_qubit(q, temp_layout) for q in virt_qargs]
        possible_swaps = []
        extra_swaps = []

        swap = {'swap': None, 'score': None}
        for q in self._coupling_graph[qubits[0]]:
            swap['swap'], swap['score'] = self.score_swap([qubits[0], q], temp_layout, to_execute,
                                                          next_gates=self._next_gates)
            extra_swaps.append(swap)
        for q in self._coupling_graph[qubits[1]]:
            swap['swap'], swap['score'] = self.score_swap([qubits[1], q], temp_layout, to_execute,
                                                          next_gates=self._next_gates)
            extra_swaps.append(swap)

        # most reliab qubits[0] to qubits[1]
        right = self.swap_paths[qubits[1]][qubits[0]]
        swap['swap'], swap['score'] = self.score_swap([qubits[0], right], temp_layout, to_execute,
                                                      next_gates=self._next_gates)
        possible_swaps.append(swap)
        extra_swaps.remove(swap)
        n -= 1
        if n == 0:
            return possible_swaps

        # most reliab qubits[1] to qubits[0]
        right = self.swap_paths[qubits[0]][qubits[1]]
        swap['swap'], swap['score'] = self.score_swap([qubits[1], right], temp_layout, to_execute,
                                                      next_gates=self._next_gates)
        possible_swaps.append(swap)
        extra_swaps.remove(swap)
        n -= 1
        if n == 0:
            return sorted(possible_swaps, key=lambda x: x['score'], reverse=True)

        # shortest path if not already in possible_swaps
        shortest_path = self._coupling_map.shortest_undirected_path(qubits[0], qubits[1])
        swap['swap'], swap['score'] = self.score_swap([qubits[0], shortest_path[1]], temp_layout, to_execute,
                                                      next_gates=self._next_gates)
        if swap not in possible_swaps:
            possible_swaps.append(swap)
            extra_swaps.remove(swap)
            n -= 1
            if n == 0:
                return sorted(possible_swaps, key=lambda x: x['score'], reverse=True)

        swap['swap'], swap['score'] = self.score_swap([qubits[1], shortest_path[-2]], temp_layout, to_execute,
                                                      next_gates=self._next_gates)
        if swap not in possible_swaps:
            possible_swaps.append(swap)
            extra_swaps.remove(swap)
            n -= 1
            if n == 0:
                return sorted(possible_swaps, key=lambda x: x['score'], reverse=True)

        while n != 0 and extra_swaps:
            possible_swaps.append(extra_swaps[0])
            extra_swaps.remove(extra_swaps[0])
            n -= 1

        return sorted(possible_swaps, key=lambda x: x['score'], reverse=True)

    def score_swap(self, swap, layout, to_execute, next_gates=10, to_map=None):
        return self.score_swap_alpha(swap, layout, to_execute, next_gates, to_map)

    def score_swap_alpha(self, swap, layout, to_execute, next_gates=5, to_map=None):
        temp_layout = layout.copy()
        temp_layout.swap(*swap)
        reliabs = list()
        if to_map:
            reliabs.extend([self.swap_reliabs[self.get_phys_qubit(gate.qargs[0], temp_layout)]
                         [self.get_phys_qubit(gate.qargs[1], temp_layout)]
                         for gate in to_map])
        reliabs.extend([self.swap_reliabs[self.get_phys_qubit(gate.qargs[0], temp_layout)]
                         [self.get_phys_qubit(gate.qargs[1], temp_layout)]
                         for gate in to_execute[:next_gates]
                         if gate.name not in ["barrier", "snapshot", "save", "load", "noise"] and \
                         len(gate.qargs) == 2])
        reliab = reduce(lambda x, y: x + y, reliabs)
        reliab /= len(reliabs)

        distance = 0
        count = 0
        if to_map:
            for gate in to_map:
                if not self._coupling_map.distance(self.get_phys_qubit(gate.qargs[0], temp_layout),
                                               self.get_phys_qubit(gate.qargs[1], temp_layout)) == 1:
                    distance += (self._coupling_map.distance(self.get_phys_qubit(gate.qargs[0], temp_layout),
                                               self.get_phys_qubit(gate.qargs[1], temp_layout))-1)/(self._max_distance-1)
                count += 1
        for gate in to_execute:
            if gate.name not in ["barrier", "snapshot", "save", "load", "noise"] and len(gate.qargs) == 2:

                if not self._coupling_map.distance(self.get_phys_qubit(gate.qargs[0], temp_layout),
                                               self.get_phys_qubit(gate.qargs[1], temp_layout)) == 1:
                    distance += (self._coupling_map.distance(self.get_phys_qubit(gate.qargs[0], temp_layout),
                                               self.get_phys_qubit(gate.qargs[1], temp_layout))-1)/(self._max_distance-1)
                next_gates -= 1
                count += 1
                if next_gates == 0:
                    break
        distance /= count

        score = self._alpha*reliab + (1-self._alpha)*(1-distance)

        return swap, score

    def execute_gate(self, gate, layout):
        """

        Args:
            gate (DAGNode): gate to be executed.
            layout (Layout): current circuit layout.

        Returns:
            executed_gate (DAGNode): excuted gate.
        """
        executed_gate = deepcopy(gate)
        new_qargs = [self.get_reg(q, layout) for q in executed_gate.qargs]
        executed_gate.qargs = executed_gate.op.qargs = new_qargs

        return executed_gate

    def get_reg(self, virt_qubit, layout):
        """

        Args:
            virt_qubit (qiskit.circuit.Qubit): virtual qubit.
            layout (Layout): current circuit layout.

        Returns:
            current_qubit (qiskit.circuit.Qubit): virt_qubit register in the circuit.
        """

        return self._qreg[layout[virt_qubit]]

    @staticmethod
    def get_phys_qubit(virt_qubit, layout):
        """

        Args:
            virt_qubit (qiskit.circuit.Qubit): virtual qubit.
            layout (Layout): current circuit layout.

        Returns:
            physical_qubit: virt_qubit position in the coupling map.
        """

        return layout[virt_qubit]
