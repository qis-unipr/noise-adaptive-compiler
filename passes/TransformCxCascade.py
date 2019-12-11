import logging


from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions import CnotGate, U2Gate
from qiskit.transpiler import TransformationPass, TranspilerError
from qiskit.transpiler.passes import Unroller, Optimize1qGates, CXCancellation
from qiskit.qasm import pi

logger = logging.getLogger(__name__)


class TransformCxCascade(TransformationPass):
    """
    Finds CNOT cascades int the dag and transform them into nearest-neighbor CNOT sequences,
    which are more easily mapped over a real device coupling map::

        ---x--x--x--x---           --------x--------
           |  |  |  |                      |
        ---o--|--|--|---           ------x-o-x------
              |  |  |                    |   |
        ------o--|--|---    --->   ----x-o---o-x----
                 |  |                  |       |
        ---------o--|---           --x-o-------o-x--
                    |                |           |
        ------------o---           --o-----------o--


        ---o--o--o--o---           -H-------x-------H-
           |  |  |  |                       |
        ---x--|--|--|---           -H-----x-o-x-----H-
              |  |  |                     |   |
        ------x--|--|---    --->   -H---x-o---o-x---H-
                 |  |                   |       |
        ---------x--|---           -H-x-o-------o-x-H-
                    |                 |           |
        ------------x---           -H-o-----------o-H-

    """

    def __init__(self):
        """TransformCxCascade initializer.
        Raises:
            TranspilerError: if run after the layout has been set.
        """
        super().__init__()
        if self.property_set['layout']:
            raise TranspilerError('TransformCxCascade pass must be run before any layout has been set.')
        self.requires.append(Unroller(['u1', 'u2', 'u3', 'cx']))
        self._num_qubits = None
        self._wires_to_id = {}
        self._id_to_wires = {}
        self._layers = None
        self._extra_layers = None
        self._skip = []

    def run(self, dag):
        """
        Run the CNOTCascadesTransform pass over a dag circuit.
        After the transformation, proceeds to check for possible one-qubit gates optimizations and CNOT cancellations,
        as subsequent CNOT nearest-neighbor sequences could create the opportunity for useful circuit simplifications.

        Args:
            dag (DAGCircuit): the dag circuit to be searched for CNOT cascades.

        Returns:
            new_dag (DAGCircuit): a new dag where all CNOT cascades have been transformed.
        """
        # prepare new dag
        new_dag = DAGCircuit()

        new_dag.name = dag.name
        self._num_qubits = dag.num_qubits()
        for q_reg in dag.qregs.values():
            new_dag.add_qreg(q_reg)
        for c_reg in dag.cregs.values():
            new_dag.add_creg(c_reg)

        i = 0
        for q_reg in dag.qregs.values():
            for q in q_reg:
                self._wires_to_id[q] = i
                self._id_to_wires[i] = q
                i += 1

        depth = new_dag.depth()
        while True:
            new_dag = Optimize1qGates().run(new_dag)
            new_dag = CXCancellation().run(new_dag)
            new_depth = new_dag.depth()
            if new_depth < depth:
                depth = new_depth
            else:
                break

        # get dag layers
        self._layers = [layer['graph'] for layer in dag.layers()]
        # this is the list of new layers for the nearest-neighbor CNOT sequences
        self._extra_layers = {l: [] for l in range(len(self._layers))}
        # loop through all layers
        for i, layer in enumerate(self._layers):
            if i != 0:
                # add nearest-neighbor CNOT sequences in the right layer
                for gate in self._extra_layers[i-1]:
                    new_dag.apply_operation_back(*gate)

            # check all gates in the layer
            for gate in layer.op_nodes():
                temp = None
                # do not add gates that have been used in the transformation process
                if gate in self._skip:
                    continue
                # every cnot could be the starting point for a CNOT cascade
                elif gate.name == 'cx':
                    logger.debug('Check Cascade %s with qargs: %s\n' % (gate.name, gate.qargs))
                    # check for a CNOT cascade
                    temp = self.check_cascade(gate, i)
                    if temp is not None:
                        logger.info('Cascade Starts at %s with qargs: %s\n' % (gate.name, gate.qargs))
                        self._skip.extend(temp)
                    else:
                        logger.debug(
                            'Check Inverse Cascade at %s with qargs: %s\n' % (gate.name, gate.qargs))
                        # check for an inverted CNOT cascade
                        temp = self.check_inverse_cascade(gate, i)
                        if temp is not None:
                            logger.info(
                                'Inverse Cascade Starts at %s with qargs: %s\n' % (gate.name, gate.qargs))
                            self._skip.extend(temp)
                        else:
                            # apply the CNOT if no cascade was found
                            self._skip.append(gate)
                            logger.debug(
                                'Found Nothing at %s with qargs: %s\n' % (gate.name, gate.qargs))
                            new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs, gate.condition)
                else:
                    self._skip.append(gate)
                    new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs, gate.condition)
        logger.debug('Cascades found: %s' % str(self._extra_layers))

        # optimize dag after transformation
        depth = new_dag.depth()
        while True:
            new_dag = Optimize1qGates().run(new_dag)
            new_dag = CXCancellation().run(new_dag)
            new_depth = new_dag.depth()
            if new_depth < depth:
                depth = new_depth
            else:
                break

        return new_dag

    def check_cascade(self, gate, layer_id):
        """Searches for a CNOT cascade, a sequence of CNOT gates where the target qubit is the same for every CNOT
        while the control changes, and transforms it into a nearest-neighbor CNOT sequence ::

            ---x--x--x--x---           --------x--------
               |  |  |  |                      |
            ---o--|--|--|---           ------x-o-x------
                  |  |  |                    |   |
            ------o--|--|---    --->   ----x-o---o-x----
                     |  |                  |       |
            ---------o--|---           --x-o-------o-x--
                        |                |           |
            ------------o---           --o-----------o--

        Args:
            gate (DAGNode): first CNOT of a possible CNOT cascade.
            layer_id (int): layer index of the CNOT.

        Returns:
            skip (list): list of gates to be skipped as part of the CNOT cascade,
            may include one-qubit gates that appears before or after the cascade.
        """
        target = self._wires_to_id[gate.qargs[1]]
        control = self._wires_to_id[gate.qargs[0]]
        controls = [control]
        skip = [gate]
        # qubits already added to the CNOT sequence
        used = set()
        used.add(target)
        used.add(control)
        # qubits that cannot be used anymore
        off_limits = set()
        before = {}
        after = []

        # flag to identify the direction of the cascade
        descending = False
        if control > target:
            descending = True

        count = 0
        last_layer = layer_id

        double_break = False
        # loop through layers until a max limit is reached
        while count < min([2 * (self._num_qubits - 1), len(self._layers) - layer_id]):
            for gate in self._layers[layer_id + count].op_nodes():
                logger.debug('Last layer: %d' % last_layer)
                logger.debug('Layer: %d' % (layer_id + count))
                logger.debug('Off limits: %s' % off_limits)
                logger.debug('Gate Name: %s\tType: %s\tQargs: %s\tCargs: %s\tCond: %s' % (
                    gate.name, gate.type, gate.qargs, gate.cargs, gate.condition))
                if gate in self._skip:
                    for qarg in gate.qargs:
                        if self._wires_to_id[qarg] == target:
                            double_break = True
                            break
                elif gate not in skip:
                    if gate.name == 'cx' and gate not in self._skip:
                        g_control = self._wires_to_id[gate.qargs[0]]
                        g_target = self._wires_to_id[gate.qargs[1]]
                        logger.debug('Check CNOT Name: %s\tType: %s\tQargs: %s\tCargs: %s\tCond: %s' % (
                            gate.name, gate.type, [g_control, g_target], gate.cargs, gate.condition))
                        if g_control == target:
                            double_break = True
                            break
                        if g_control in off_limits or g_target in off_limits:
                            off_limits.add(g_control)
                            off_limits.add(g_target)
                            if g_control not in used:
                                used.add(g_control)
                            if g_target not in used:
                                used.add(g_target)
                            logger.debug('CNOT Off limits')
                            continue
                        logger.debug('Used: %s' % str(used))
                        logger.debug('Controls: %s' % str(controls))
                        logger.debug('Control-G_control: %d-%d' % (control, g_control))
                        logger.debug('Target-G_target: %d-%d' % (target, g_target))
                        # chek that the CNOT is part of the cascade
                        a = (g_target == target and g_control not in controls and g_control not in used)
                        b = (descending is True and g_control > target) or (descending is False and g_control < target)
                        logger.debug('A: %s B: %s Descending: %s\n' % (a, b, descending))
                        if a and b:
                            controls.append(g_control)
                            used.add(g_control)
                            skip.append(gate)
                        # check if the CNOT interrupts the cascade
                        elif g_target != target and g_control != target:
                            # remember to put the CNOT after the transformation
                            if g_target not in used and g_control not in used:
                                if last_layer < layer_id + count:
                                    last_layer = layer_id + count
                            # updates used and off limits qubits when necessary
                            else:
                                off_limits.add(g_control)
                                off_limits.add(g_target)
                                if last_layer > layer_id + count - 1:
                                    last_layer = layer_id + count - 1
                                if g_control not in used:
                                    used.add(g_control)
                                if g_target not in used:
                                    used.add(g_target)
                        else:
                            # break the loop if the CNOT interrupts the cascade
                            double_break = True
                            break
                    else:
                        # ignore gates acting on off limits qubits
                        double_continue = False
                        for qarg in gate.qargs:
                            if self._wires_to_id[qarg] in off_limits:
                                double_continue = True
                                continue
                        if double_continue is True:
                            continue

                        # for special multi-qubits gates, update used and off limits qubits properly,
                        # break the loop if necessary
                        if gate.name in ["barrier", "snapshot", "save", "load", "noise"]:
                            qargs = [self._wires_to_id[qarg] for qarg in gate.qargs]
                            if target in qargs:
                                if last_layer > layer_id + count - 1:
                                    last_layer = layer_id + count - 1
                                double_break = True
                                break
                            u = []
                            not_u = []
                            for qarg in qargs:
                                if qarg in used:
                                    off_limits.add(qarg)
                                    u.append(qarg)
                                else:
                                    not_u.append(qarg)
                            if len(u) == len(qargs):
                                # the transformation must be applied before this gate
                                if last_layer > layer_id + count - 1:
                                    last_layer = layer_id + count - 1
                            elif len(u) == 0:
                                # the transformation must be applied after this gate
                                if last_layer < layer_id + count:
                                    last_layer = layer_id + count
                            else:
                                # the transformation must be applied before this gate
                                if last_layer > layer_id + count - 1:
                                    last_layer = layer_id + count - 1
                                for qarg in not_u+u:
                                    used.add(qarg)
                                    off_limits.add(qarg)
                        else:
                            # check if one-qubits gates either interrupt the cascade, can be applied after or before
                            qarg = self._wires_to_id[gate.qargs[0]]
                            logger.debug(gate.op.__class__)
                            logger.debug('Gate Name: %s\tType: %s\tQarg: %s\tCarg: %s\tCond: %s' % (
                                gate.name, gate.type, qarg, gate.cargs, gate.condition))
                            if qarg == target:
                                logger.debug('After')
                                after.append(gate)
                                skip.append(gate)
                                double_break = True
                                break
                            if qarg not in used:
                                logger.debug('Before')
                                if qarg not in before:
                                    before[qarg] = []
                                before[qarg].append(gate)
                                skip.append(gate)
                            else:
                                logger.debug('After')
                                after.append(gate)
                                skip.append(gate)
            count += 1
            if double_break is True:
                break
        # if a cascade was found
        if len(controls) > 1:
            logger.debug('Found Cascade from layer %d to %d\n' % (layer_id, last_layer))
            if descending is True:
                controls = sorted(controls)
            else:
                controls = sorted(controls, reverse=True)

            # apply all gates that were encountered before the cascade
            for u in before:
                for gate in before[u]:
                    self._extra_layers[last_layer].append(
                        (gate.op.__class__(*gate.op.params), gate.qargs, gate.cargs, gate.condition))

            # apply the transformation
            for i in range(len(controls) - 1, 0, -1):
                self._extra_layers[last_layer].append((CnotGate(),
                                                       [self._id_to_wires[controls[i]],
                                                        self._id_to_wires[controls[i - 1]]], []))
            self._extra_layers[last_layer].append(
                (CnotGate(), [self._id_to_wires[controls[0]], self._id_to_wires[target]], []))
            for i in range(len(controls) - 1):
                self._extra_layers[last_layer].append((CnotGate(),
                                                       [self._id_to_wires[controls[i + 1]],
                                                        self._id_to_wires[controls[i]]], []))

            # apply all gates that were encountered after the cascade
            for gate in after:
                self._extra_layers[last_layer].append(
                    (gate.op.__class__(*gate.op.params), gate.qargs, gate.cargs, gate.condition))
        else:
            skip = None

        return skip

    def check_inverse_cascade(self, gate, layer_id):
        """Searches for an inverted CNOT cascade, a sequence of CNOT gates where the control qubit is the same for every CNOT
        while the target changes, and transforms it into a nearest-neighbor CNOT sequence.
         It is very similar to a CNOT cascade by using H gates to invert every CNOT.
         For every H gate it adds another H gate to maintain the circuit identity::

            ---o--o--o--o---           -H-------x-------H-
               |  |  |  |                       |
            ---x--|--|--|---           -H-----x-o-x-----H-
                  |  |  |                     |   |
            ------x--|--|---    --->   -H---x-o---o-x---H-
                     |  |                   |       |
            ---------x--|---           -H-x-o-------o-x-H-
                        |                 |           |
            ------------x---           -H-o-----------o-H-

        Args:
            gate (DAGNode): first CNOT of a possible inverted CNOT cascade.
            layer_id (int): layer index of the CNOT.

        Returns:
            skip (list): list of gates to be skipped as part of the inverted CNOT cascade,
            may include one-qubit gates that appears before or after the cascade.
        """
        target = self._wires_to_id[gate.qargs[1]]
        control = self._wires_to_id[gate.qargs[0]]
        targets = [target]
        skip = [gate]
        # qubits already added to the CNOT sequence
        used = set()
        used.add(target)
        used.add(control)
        # qubits that cannot be used anymore
        off_limits = set()
        before = {}
        after = []

        # flag to identify the direction of the cascade
        descending = False
        if target > control:
            descending = True

        count = 0
        last_layer = layer_id

        double_break = False
        # loop through layers until a max limit is reached
        while count < min([2 * (self._num_qubits-1), len(self._layers)-layer_id]):
            for gate in self._layers[layer_id + count].op_nodes():
                logger.debug('Last layer: %d' % last_layer)
                logger.debug('Layer: %d' % (layer_id+count))
                logger.debug('Off limits: %s' % off_limits)
                logger.debug('Gate Name: %s\tType: %s\tQargs: %s\tCargs: %s\tCond: %s' % (
                    gate.name, gate.type, gate.qargs, gate.cargs, gate.condition))
                if gate in self._skip:
                    for qarg in gate.qargs:
                        if self._wires_to_id[qarg] == control:
                            double_break = True
                            break
                elif gate not in skip:
                    if gate.name == 'cx' and gate not in self._skip:
                        g_control = self._wires_to_id[gate.qargs[0]]
                        g_target = self._wires_to_id[gate.qargs[1]]
                        logger.debug('Check CNOT Name: %s\tType: %s\tQargs: %s\tCargs: %s\tCond: %s' % (
                            gate.name, gate.type, [g_control, g_target], gate.cargs, gate.condition))
                        if g_target == control:
                            double_break = True
                            break
                        if g_control in off_limits or g_target in off_limits:
                            if last_layer > layer_id + count - 1:
                                last_layer = layer_id + count - 1
                            off_limits.add(g_control)
                            off_limits.add(g_target)
                            if g_control not in used:
                                used.add(g_control)
                            if g_target not in used:
                                used.add(g_target)
                            logger.debug('CNOT off limits')
                            continue
                        logger.debug('Used: %s' % str(used))
                        logger.debug('Targets: %s' % str(targets))
                        logger.debug('Control-G_control: %d-%d' % (control, g_control))
                        logger.debug('Target-G_target: %d-%d' % (target, g_target))
                        # chek that the CNOT is part of the cascade
                        a = (g_control == control and g_target not in targets and g_target not in used)
                        b = (descending is True and g_target > control) or (descending is False and g_target < control)
                        logger.debug('A: %s B: %s Descending: %s\n' % (a, b, descending))
                        if a and b:
                            targets.append(g_target)
                            used.add(g_target)
                            skip.append(gate)
                        # check if the CNOT interrupts the cascade
                        elif g_control != control and g_target != control:
                            # remember to put the CNOT after the transformation
                            if g_control not in used and g_target not in used:
                                if last_layer < layer_id+count:
                                    last_layer = layer_id+count
                            # updates used and off limits qubits when necessary
                            else:
                                off_limits.add(g_control)
                                off_limits.add(g_target)
                                if last_layer > layer_id+count-1:
                                    last_layer = layer_id+count-1
                                if g_control not in used:
                                    used.add(g_control)
                                if g_target not in used:
                                    used.add(g_target)
                        else:
                            # break the loop if the CNOT interrupts the cascade
                            double_break = True
                            break
                    else:
                        # ignore gates acting on off limits qubits
                        double_continue = False
                        for qarg in gate.qargs:
                            if self._wires_to_id[qarg] in off_limits:
                                double_continue = True
                                continue
                        if double_continue is True:
                            continue

                        # for special multi-qubits gates, update used and off limits qubits properly,
                        # break the loop if necessary
                        if gate.name in ["barrier", "snapshot", "save", "load", "noise"]:
                            qargs = [self._wires_to_id[qarg] for qarg in gate.qargs]
                            if control in qargs:
                                if last_layer > layer_id + count-1:
                                    last_layer = layer_id + count-1
                                double_break = True
                                break
                            u = []
                            not_u = []
                            for qarg in qargs:
                                if qarg in used:
                                    off_limits.add(qarg)
                                    u.append(qarg)
                                else:
                                    not_u.append(qarg)
                            if len(u) == len(qargs):
                                # the transformation must be applied before this gate
                                if last_layer > layer_id + count-1:
                                    last_layer = layer_id + count-1
                            elif len(u) == 0:
                                # the transformation must be applied after this gate
                                if last_layer < layer_id + count:
                                    last_layer = layer_id + count
                            else:
                                # the transformation must be applied before this gate
                                if last_layer > layer_id + count-1:
                                    last_layer = layer_id + count-1
                                for qarg in not_u+u:
                                    used.add(qarg)
                                    off_limits.add(qarg)
                        else:
                            # check if one-qubits gates either interrupt the cascade, can be applied after or before
                            qarg = self._wires_to_id[gate.qargs[0]]
                            logger.debug(gate.op.__class__)
                            logger.debug('Gate Name: %s\tType: %s\tQarg: %s\tCarg: %s\tCond: %s' % (
                                gate.name, gate.type, qarg, gate.cargs, gate.condition))
                            if qarg == control:
                                logger.debug('After')
                                after.append(gate)
                                skip.append(gate)
                                double_break = True
                                break
                            if qarg not in used:
                                logger.debug('Before')
                                if qarg not in before:
                                    before[qarg] = []
                                before[qarg].append(gate)
                                skip.append(gate)
                            else:
                                logger.debug('After')
                                after.append(gate)
                                skip.append(gate)

            count += 1
            if double_break is True:
                break
        # if an inverse cascade was found
        if len(targets) > 1:
            logger.debug('Found Inverse Cascade from layer %d to %d\n' % (layer_id, last_layer))
            if descending is True:
                targets = sorted(targets)
            else:
                targets = sorted(targets, reverse=True)

            # apply all gates that were encountered before the cascade
            for u in before:
                for gate in before[u]:
                    self._extra_layers[last_layer].append((gate.op.__class__(*gate.op.params), gate.qargs, gate.cargs, gate.condition))

            # apply the transformation
            self._extra_layers[last_layer].append((U2Gate(0, pi), [self._id_to_wires[control]], []))
            for t in targets:
                self._extra_layers[last_layer].append((U2Gate(0, pi), [self._id_to_wires[t]], []))
            for i in range(len(targets) - 1, 0, -1):
                self._extra_layers[last_layer].append((CnotGate(),
                                                                 [self._id_to_wires[targets[i]],
                                                                  self._id_to_wires[targets[i - 1]]], []))
            self._extra_layers[last_layer].append(
                (CnotGate(), [self._id_to_wires[targets[0]], self._id_to_wires[control]], []))
            for i in range(len(targets) - 1):
                self._extra_layers[last_layer].append((CnotGate(),
                                                                 [self._id_to_wires[targets[i + 1]],
                                                                  self._id_to_wires[targets[i]]], []))
            self._extra_layers[last_layer].append((U2Gate(0, pi), [self._id_to_wires[control]], []))
            for t in targets:
                self._extra_layers[last_layer].append((U2Gate(0, pi), [self._id_to_wires[t]], []))

            # apply all gates that were encountered after the cascade
            for gate in after:
                self._extra_layers[last_layer].append((gate.op.__class__(*gate.op.params), gate.qargs, gate.cargs, gate.condition))
        else:
            skip = None

        return skip
