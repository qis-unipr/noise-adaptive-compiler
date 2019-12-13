import logging

from networkx import shortest_path

from qiskit.transpiler import AnalysisPass, TranspilerError, CouplingMap, Layout

logger = logging.getLogger(__name__)


class ChainLayout(AnalysisPass):
    """
    Maps the qubits in the coupling map to a nearest-neighbor sequence of qubits,
    called *chain*, to be used as an initial layout.
    Sometimes not all qubits in a device can be arranged in a chain.
    If necessary, such outliers will be inserted in the chain after one of their neighbors.
    """

    def __init__(self, coupling_map, backend_prop=None):
        """ChainLayout initializer.

        Args:
            coupling_map (CouplingMap or list): directed graph representing a coupling chain.
            backend_prop (BackendProperties): backend properties object.
        Raises:
            TranspilerError: if invalid options.
        """
        super().__init__()
        if isinstance(coupling_map, list):
            self.coupling_map = CouplingMap(coupling_map)
        elif isinstance(coupling_map, CouplingMap):
            self.coupling_map = coupling_map
        else:
            raise TranspilerError('Coupling map of type %s is not valid' % coupling_map.__class__)

        self.coupling_graph = self.coupling_map.graph.to_undirected()

        self.backend_prop = backend_prop
        # collect cx reliability data
        if self.backend_prop is not None:
            self.cx_reliab = dict()
            for ginfo in self.backend_prop.gates:
                if ginfo.gate == 'cx':
                    for item in ginfo.parameters:
                        if item.name == 'gate_error':
                            g_reliab = 1.0 - item.value
                            break
                        else:
                            g_reliab = 1.0
                    self.cx_reliab[(ginfo.qubits[0], ginfo.qubits[1])] = g_reliab
                    self.cx_reliab[(ginfo.qubits[1], ginfo.qubits[0])] = g_reliab

    def run(self, dag):
        """Sets the layout property set.

        Args:
            dag (DAGCircuit): DAG to find layout for.
        Raises:
            TranspilerError: if dag wider than the coupling_map.
        """

        num_dag_qubits = sum([qreg.size for qreg in dag.qregs.values()])
        if num_dag_qubits > self.coupling_map.size():
            raise TranspilerError('Number of qubits greater than device.')
        # get the chain of qubits as list of integers
        chain = self.chain(num_dag_qubits)
        layout = Layout()
        chain_iter = 0
        # produce a layout from the chain
        for qreg in dag.qregs.values():
            for i in range(qreg.size):
                layout[qreg[i]] = chain[chain_iter]
                chain_iter += 1
        self.property_set['layout'] = layout
        logger.info(self.property_set['layout'])

    def chain(self, num_qubits=None):
        """Finds a chain  of qubits such that qubit *i* has a connection
        with qubits *(i-1)* and *(i+1)* in the coupling chain.
        Relies on best_subset() to select a subset of qubits with high cx reliability.
        Sometimes not all qubits in a device can be arranged in a chain.
        If necessary, such outliers will be inserted in the chain after one of their neighbors.

        Args:
            num_qubits (int): number of virtual qubits,
                defaults to the number of qubits of the coupling chain.
        Raises:
            TranspilerError: if invalid options
        """

        if num_qubits is None:
            num_qubits = self.coupling_map.size()
        if num_qubits > self.coupling_map.size():
            raise TranspilerError('Number of qubits greater than device.')

        current = 0
        full_map = [current]
        isolated = []
        isolated_with_data = []
        explored = set()
        explored.add(current)

        last_back_step = None
        # loop over the coupling map until all qubits no more qubits
        # can be connected to the chain
        while True:
            neighbors = []
            no_neighbors = True
            for n in self.coupling_graph[current].keys():
                if n not in explored:
                    no_neighbors = False
                    neighbors.append(n)
            logger.debug('Neighbors: %s' % str(neighbors))
            # try to select next qubit from neighbors of last connected qubit
            if no_neighbors is False:
                if current + 1 in neighbors:
                    next = current + 1
                else:
                    next = min(neighbors)
            else:
                # if no neighbors are found, go back the chain until a new neighbor is found
                # and restart the loop from there
                if self.backend_prop is None:
                    isolated_with_data.append((full_map[-2], current))
                else:
                    isolated_with_data.append(
                        (full_map[-2], current, self.cx_reliab[(full_map[-2], current)]))
                isolated.append(current)
                full_map.remove(current)
                current = full_map[-1]
                logger.debug('last back step: %s' % str(last_back_step))
                if current != last_back_step:
                    last_back_step = current
                else:
                    break
                continue

            explored.add(next)
            current = next
            full_map.append(next)

            logger.debug('Full chain: %s' % str(full_map))
            logger.debug('Explored: %s' % str(explored))
            logger.debug('Isolated: %s' % str(isolated_with_data))

            # check that there are still qubits to explore
            if len(explored) < self.coupling_map.size() - 1:
                neighbors1 = []
                for n1 in self.coupling_graph[next].keys():
                    if n1 not in explored:
                        neighbors1.append(n1)

                # check that the selected qubit does not lead to a dead end
                for n1 in neighbors1:
                    to_remove = True
                    for n2 in self.coupling_graph[n1].keys():
                        if n2 not in explored or n2 == next:
                            to_remove = False
                    if to_remove is True:
                        explored.add(n1)
                        if self.backend_prop is None:
                            isolated_with_data.append((next, n1))
                        else:
                            isolated_with_data.append((next, n1, self.cx_reliab[(next, n1)]))
                        isolated.append(n1)

            # break the loop when all qubits have been explored
            if len(explored) == self.coupling_map.size():
                # if enough qubits have been connected,
                # return a subset with high cx reliability
                if len(full_map) >= num_qubits:
                    return self.best_subset(full_map, num_qubits)
                break

        # check for isolated qubits
        for q in range(self.coupling_map.size()):
            if q not in explored and q not in isolated:
                for i in isolated:
                    if q in self.coupling_graph[i].keys():
                        if self.backend_prop is None:
                            isolated_with_data.append((i, q))
                        else:
                            isolated_with_data.append((i, q, self.cx_reliab[(i, q)]))
                        isolated.append(q)
                        explored.add(q)
                        break
                if q not in isolated:
                    for n in self.coupling_graph[q].keys():
                        if n in full_map:
                            if self.backend_prop is None:
                                isolated_with_data.append((n, q))
                            else:
                                isolated_with_data.append((n, q, self.cx_reliab[(n, q)]))
                            isolated.append(q)
                            explored.add(q)
                            break

        # if the chain is not long enough, add the isolated qubits
        logger.debug('Searching for isolated')
        remaining = num_qubits - len(full_map)
        if remaining > 0:
            if self.backend_prop is not None:
                isolated_with_data = sorted(isolated_with_data, key=lambda x: x[2], reverse=True)
            while remaining > 0:
                for next in isolated_with_data:
                    if next[0] in full_map:
                        logger.debug(next)
                        full_map.insert(full_map.index(next[0]) + 1, next[1])
                        isolated_with_data.remove(next)
                        isolated.remove(next[1])
                        remaining -= 1
                        break
        return full_map

    def best_subset(self, chain, num_qubits):
        """Selects from the chain a subset of qubits with high cx reliability.

        Args:
            chain (list): a chain of qubits.
            num_qubits (int): dimension of the subset.

        Returns:
            best_subset (list): subset with high cx reliability.
        """
        if self.backend_prop is None:
            best_reliab = float('inf')
        else:
            best_reliab = 0
        best = None
        # use a moving window over the chain to select a subset with high cx reliability
        # if no backend information are provided, use the distance between qubits as a metric
        for offset in range(len(chain) - num_qubits):
            sub_set = chain[offset:offset + num_qubits]
            if self.backend_prop is None:
                tot_reliab = 0
            else:
                tot_reliab = 1
            for q in range(len(sub_set) - 1):
                if sub_set[q + 1] not in self.coupling_graph[q].keys():
                    path = shortest_path(self.coupling_graph, source=sub_set[q], target=sub_set[q + 1])
                    for p in range(len(path) - 1):
                        if self.backend_prop is None:
                            tot_reliab += 1
                        else:
                            tot_reliab *= self.cx_reliab[(path[p], path[p + 1])] ** 3
                else:
                    if self.backend_prop is None:
                        tot_reliab += 1
                    else:
                        tot_reliab *= self.cx_reliab[(sub_set[q], sub_set[q + 1])]
            if self.backend_prop is None:
                if tot_reliab < best_reliab:
                    best_reliab = tot_reliab
                    best = sub_set
            else:
                if tot_reliab > best_reliab:
                    best_reliab = tot_reliab
                    best = sub_set
        return best
