from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, UCC, HartreeFock, PUCCD, SUCCD

from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper, ParityMapper

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit

from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit.algorithms.gradients import FiniteDiffEstimatorGradient
from qiskit.algorithms.optimizers import L_BFGS_B, SPSA, SLSQP, NELDER_MEAD
from qiskit.algorithms.optimizers.spsa import powerseries
from qiskit.primitives import Estimator
from qiskit.utils import algorithm_globals

from qiskit import *
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import TaperedPauliSumOp

from qiskit_ibm_runtime import (QiskitRuntimeService, Session, Estimator as
                                RuntimeEstimator)

import numpy as np
import mapomatic as mm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import importlib

seed = 170
algorithm_globals.random_seed = seed


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def construct_problem(geometry,
                      charge,
                      spin,
                      basis,
                      mapper_type,
                      num_spatial_orbitals,
                      num_electrons=None,
                      z2symmetry_reduction=None):

    # The spin of the molecule. In accordance with PySCF’s definition, the spin equals 2*S,
    # where S is the total spin number of the molecule.
    driver = PySCFDriver(
        atom=geometry,
        basis=basis,
        charge=charge,
        spin=int(2 * spin),
        unit=DistanceUnit.ANGSTROM,
    )

    # Run the preliminary quantum chemistry calculation
    problem = driver.run()

    if num_electrons is None:
        num_electrons = problem.num_particles

    # Set the active space
    # Define the active space around the Fermi level
    # (selected automatically around the HOMO and LUMO, ordered by energy)
    transformer = ActiveSpaceTransformer(
        num_electrons=num_electrons,  # Number of electrons in our active space
        num_spatial_orbitals=
        num_spatial_orbitals,  # Number of orbitals in our active space
        active_orbitals=None  # May be useful
    )

    # Now you can get the reduced electronic structure problem
    problem_reduced = transformer.transform(problem)

    # The second quantized Hamiltonian of the reduce problem
    second_q_ops_reduced = problem_reduced.second_q_ops()

    # Setup the mapper and qubit converter
    if mapper_type == 'ParityMapper':
        mapper = ParityMapper()
    elif mapper_type == 'JordanWignerMapper':
        mapper = JordanWignerMapper()
    elif mapper_type == 'BravyiKitaevMapper':
        mapper = BravyiKitaevMapper()

    # Set the qubit converter with two qubit reduction to reduce the computational cost
    converter = QubitConverter(mapper,
                               two_qubit_reduction=True,
                               z2symmetry_reduction=z2symmetry_reduction)

    # Compute the Hamitonian in qubit form
    qubit_op = converter.convert(
        second_q_ops_reduced[0],
        num_particles=problem_reduced.num_particles,
        sector_locator=problem_reduced.symmetry_sector_locator)

    # Get reference solution
    numpy_solver = NumPyMinimumEigensolver()
    calc = GroundStateEigensolver(converter, numpy_solver)
    res_actual = calc.solve(problem_reduced)

    return problem_reduced, converter, qubit_op, res_actual


def get_ansatz(ansatz_type, reps, problem_reduced, converter, qubit_op):

    # Hartree focks state
    num_particles = problem_reduced.num_particles
    num_spatial_orbitals = problem_reduced.num_spatial_orbitals

    init_state = HartreeFock(num_spatial_orbitals=num_spatial_orbitals,
                             num_particles=num_particles,
                             qubit_converter=converter)

    # Choose the ansatz

    # Put arguments for twolocal
    if ansatz_type == "TwoLocal":
        # Single qubit rotations that are placed on all qubits with independent parameters
        rotation_blocks = ['ry']
        # Entangling gates
        entanglement_blocks = 'cx'
        # How the qubits are entangled
        entanglement = 'linear'
        # Repetitions of rotation_blocks + entanglement_blocks with independent parameters
        repetitions = reps
        # Skip the final rotation_blocks layer
        skip_final_rotation_layer = False
        ansatz = TwoLocal(qubit_op.num_qubits,
                          rotation_blocks,
                          entanglement_blocks,
                          reps=repetitions,
                          entanglement=entanglement,
                          skip_final_rotation_layer=skip_final_rotation_layer)
        # Add the initial state
        ansatz.compose(init_state, front=True, inplace=True)
    elif ansatz_type == "UCCSD":
        ansatz = UCCSD(num_spatial_orbitals=num_spatial_orbitals,
                       num_particles=num_particles,
                       qubit_converter=converter,
                       reps=reps,
                       initial_state=init_state)
    elif ansatz_type == "PUCCD":
        ansatz = PUCCD(num_spatial_orbitals=num_spatial_orbitals,
                       num_particles=num_particles,
                       qubit_converter=converter,
                       reps=reps,
                       initial_state=init_state)
    elif ansatz_type == "SUCCD":
        ansatz = SUCCD(num_spatial_orbitals=num_spatial_orbitals,
                       num_particles=num_particles,
                       qubit_converter=converter,
                       reps=reps,
                       initial_state=init_state)
    elif ansatz_type == 'UCC':
        ansatz = UCC(num_spatial_orbitals=num_spatial_orbitals,
                     num_particles=num_particles,
                     qubit_converter=converter,
                     reps=reps,
                     initial_state=init_state,
                     excitations='sd')

    return ansatz


def device_mapping(ansatz, backend, qubit_op, num_device_qubits):

    # find qubit layout
    trans_qc = transpile(ansatz,
                         backend=backend,
                         optimization_level=3,
                         seed_transpiler=seed)
    small_qc = mm.deflate_circuit(trans_qc)
    score = mm.best_overall_layout(small_qc, backend)

    # Extract the quantum retmgisters from score list
    q_regs = score[0]

    q_layout = q_regs

    # fake backend optimal ansatz
    ansatz_opt = transpile(ansatz,
                           backend=backend,
                           initial_layout=q_layout,
                           optimization_level=3,
                           seed_transpiler=seed)

    # map hamiltonian to backend
    coeff = qubit_op.coeff
    z2_symmetries = qubit_op.z2_symmetries

    ops = []
    n_qubits = num_device_qubits

    ancilla_qubits = np.setdiff1d(np.arange(0, n_qubits), q_layout)

    for op in qubit_op:
        pauli_string = op.primitive.paulis[0]

        for i in ancilla_qubits:
            pauli_string = pauli_string.insert(i, Pauli("I"))

        pauli_string = pauli_string.__str__()

        pauli_coeff = op.primitive.coeffs[0]
        ops.append((pauli_string, pauli_coeff))

    new_qubit_op = TaperedPauliSumOp(SparsePauliOp.from_list(ops),
                                     z2_symmetries, coeff)

    return ansatz_opt, new_qubit_op


def init_point_finder(ansatz_opt, optimizer_type, new_qubit_op):
    inits = np.linspace(-np.pi, np.pi, 50)
    values = []

    for i in range(len(inits)):
        try:
            initial_point = [inits[i]] * len(ansatz_opt.ordered_parameters)
        except:
            initial_point = [inits[i]] * ansatz_opt.num_parameters

        estimator = Estimator(options={
            'seed': seed,
            "seed_transpiler": seed,
            "optimization_level": 0
        })

        opt = _import_class(f"qiskit.algorithms.optimizers.{optimizer_type}")(
            maxiter=0)

        vqe = VQE(estimator, ansatz_opt, opt, initial_point=initial_point)
        result = vqe.compute_minimum_eigenvalue(operator=new_qubit_op)

        intermediate_info = []
        five_percent = []
        one_percent = []
        ev = result.eigenvalue
        values.append(ev)
        l = len(values)
        clear_output(wait=True)
        plt.ylabel('Energy')
        plt.xlabel('init_point')
        plt.plot(inits[:l],
                 values,
                 color='purple',
                 lw=2,
                 label='Simulated VQE')
        plt.legend()
        plt.grid()
        plt.show()

    idx = np.where(values <= min(values))[0]

    return inits[idx][0]


def custom_vqe(estimator,
               ansatz_opt,
               optimizer,
               new_qubit_op,
               exact_energy,
               execution='local',
               multiplier=None,
               init_point=None):

    # Define a simple callback function
    intermediate_info = []
    five_percent = []
    one_percent = []

    def callback(eval_count, parameters, value, std):
        intermediate_info.append(value)
        five_percent.append(exact_energy * (1 - 0.05))
        one_percent.append(exact_energy * (1 - 0.01))
        clear_output(wait=True)
        plt.plot(intermediate_info,
                 color='purple',
                 lw=2,
                 label=f'Simulated VQE {np.round(value,4)}')
        plt.ylabel('Energy')
        plt.xlabel('Iterations')
        # Exact ground state energy value
        plt.axhline(y=exact_energy,
                    color="tab:red",
                    ls="--",
                    lw=2,
                    label="Target: " + str(np.round(exact_energy, 4)))
        plt.plot(five_percent,
                 lw=1,
                 label=f'5% ({np.round(exact_energy*(1-0.05),4)})')
        plt.plot(one_percent,
                 lw=1,
                 label=f'1% ({np.round(exact_energy*(1-0.01),4)})')
        plt.legend()
        plt.grid()
        plt.show()

    def callback_sim(eval_count, parameters, value, std):
        intermediate_info.append(value)
        five_percent.append(exact_energy * (1 - 0.05))
        one_percent.append(exact_energy * (1 - 0.01))

    if multiplier == None:
        try:
            initial_point = np.random.uniform(
                size=(len(ansatz_opt.ordered_parameters)))
        except:
            initial_point = np.random.uniform(size=ansatz_opt.num_parameters)
    else:
        try:
            initial_point = [multiplier] * len(ansatz_opt.ordered_parameters)
        except:
            initial_point = [multiplier] * ansatz_opt.num_parameters

    if init_point is not None:
        initial_point = init_point

    if execution == 'local':
        gradient = FiniteDiffEstimatorGradient(estimator, epsilon=0.001)
        vqe = VQE(estimator,
                  ansatz_opt,
                  optimizer,
                  callback=callback,
                  gradient=gradient,
                  initial_point=initial_point)
        result = vqe.compute_minimum_eigenvalue(operator=new_qubit_op)
    else:
        service = QiskitRuntimeService(channel='ibm_quantum')
        backend = 'aer_simulator'
        with Session(service=service, backend=backend) as session:

            # Prepare primitive
            rt_estimator = RuntimeEstimator(session=session)
            # Set up algorithm

            gradient = FiniteDiffEstimatorGradient(rt_estimator, epsilon=0.001)
            vqe = VQE(rt_estimator,
                      ansatz_opt,
                      optimizer,
                      callback=callback_sim,
                      gradient=gradient,
                      initial_point=initial_point)

            # Run algorithm
            result = vqe.compute_minimum_eigenvalue(operator=new_qubit_op)

    return result, intermediate_info


def get_optimizer(optimizer_type, max_iter=50, a=0.1, c=0.1):
    maxiter = max_iter
    stability_constant = 1
    c = c
    alpha = 0.602
    gamma = 0.101

    a = a

    # set up the powerseries
    def learning_rate():
        return powerseries(a, alpha, stability_constant)

    def perturbation():
        return powerseries(c, gamma)

    if optimizer_type == 'SPSA':
        optimizer = SPSA(maxiter=maxiter,
                         learning_rate=learning_rate,
                         perturbation=perturbation)
    elif optimizer_type == 'SLSQP':
        optimizer = SLSQP(maxiter=maxiter)
    elif optimizer_type == 'NELDER_MEAD':
        optimizer = NELDER_MEAD(maxiter=maxiter)
    elif optimizer_type == 'L_BFGS_B':
        optimizer = L_BFGS_B(maxiter=maxiter)

    return optimizer


def rel_err(target, measured):
    return abs((target - measured) / target)


def deparameterise(circuit, optimal_parameters, freeze_indices, freeze_value):
    partial_params = {}
    for (i, j) in optimal_parameters.items():
        if i.index in freeze_indices:
            partial_params[i] = freeze_value

    for i, j in partial_params.items():
        print(i, j)

    return circuit.assign_parameters(partial_params)
