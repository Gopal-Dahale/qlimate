"""
Perform quantum simulation using Rigetti's stack, using packages such as pyquil and forestopenfermion.
"""

from enum import Enum
import numpy as np

from pyscf import ao2mo, scf
from openfermion.transforms import jordan_wigner
from openfermion.chem import MolecularData
import pennylane as qml
import warnings


class PennyLaneParametricSolver:
    """Performs an energy estimation for a molecule with a parametric circuit.

    Performs energy estimations for a given molecule and a choice of ansatz
    circuit that is supported.

    Uses the CCSD method to solve the electronic structure problem.
    PySCF program will be utilized.
    Users can also provide a function that takes a `pyscf.gto.Mole`
    as its first argument and `pyscf.scf.RHF` as its second.

    Attributes:
        optimized_amplitudes (list): The optimized UCCSD amplitudes.
        of_mole (openfermion.hamiltonian.MolecularData): Molecular Data in Openfermion.
        f_hamiltonian (openfermion.ops.InteractionOperator): Fermionic Hamiltonian.
        qubit_hamiltonian (openfermion.transforms.jordan_wigner): Qubit Hamiltonian.
        n_qubits (int): Number of qubits.
    """

    class Ansatze(Enum):
        """ Enumeration of the ansatz circuits that are supported."""
        UCCSD = 0

    def __init__(self,
                 ansatz,
                 molecule,
                 mean_field=None,
                 occupied=None,
                 active=None):
        """Initialize the settings for simulation.

        If the mean field is not provided it is automatically calculated.

        Args:
            ansatz (OpenFermionParametricSolver.Ansatze): Ansatz for the quantum solver.
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.
        """

        # Check the ansatz
        # assert(isinstance(ansatz, PennyLaneParametricSolver.Ansatze))
        self.ansatz = ansatz

        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()

            if not mean_field.converged:
                orb_temp = mean_field.mo_coeff
                occ_temp = mean_field.mo_occ
                nr = scf.newton(mean_field)
                energy = nr.kernel(orb_temp, occ_temp)
                mean_field = nr

        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn(
                "PennyLaneParametricSolver simulating with mean field not converged.",
                RuntimeWarning)

        self.molecule = molecule
        self.mean_field = mean_field

        # Initialize the amplitudes (parameters to be optimized)
        self.optimized_amplitudes = []

        # Initialize molecule quantities
        # ------------------------------
        self.num_orbitals = len(mean_field.mo_energy)
        self.num_spin_orbitals = self.num_orbitals * 2

        print("Number of orbitals: ", self.num_orbitals)

        # Set the parameters for Openfermion
        self.of_mole = self._build_of_molecule(molecule, mean_field)

        if (occupied is not None) and (active is not None):
            self.f_hamiltonian = self.of_mole.get_molecular_hamiltonian(
                occupied_indices=occupied, active_indices=active)
        else:
            self.f_hamiltonian = self.of_mole.get_molecular_hamiltonian()

        # Transform the fermionic Hamiltonian into qubit Hamiltonian
        qubit_hamiltonian = jordan_wigner(self.f_hamiltonian)
        qubit_hamiltonian.compress()

        self.H = qml.qchem.import_operator(qubit_hamiltonian)
        print("number of hamiltonian terms: ", len(self.H.ops))
        print("number of wires: ", len(self.H.wires))

        self.n_qubits = len(self.H.wires)

        self.amplitude_dimension = (self.n_qubits, 2)
        print("Number of params expected", self.amplitude_dimension)
        # Define the device
        dev = qml.device("lightning.qubit", wires=self.n_qubits)
        qubits = list(range(self.n_qubits))

        ############################ Tapering ############################
        # After tapering the hamiltonian, the function get_rdm may not work.
        # This is due to the fact that the function iterates over the terms
        # of fermionic hamiltonian and creates a new hamiltonian in every iteration
        # which may not contain the same number of qubits as the tapered hamiltonian.

        generators = qml.symmetry_generators(self.H)
        paulixops = qml.paulix_ops(generators, self.n_qubits)
        paulix_sector = qml.qchem.optimal_sector(self.H, generators,
                                                 molecule.nelectron)
        H_tapered = qml.taper(self.H, generators, paulixops, paulix_sector)

        print("number of hamiltonian terms (tapered): ", len(H_tapered.ops))
        print("number of wires (tapered): ", len(H_tapered.wires))

        dev = qml.device("lightning.qubit", wires=len(H_tapered.wires))
        qubits = H_tapered.wires
        self.n_qubits = len(H_tapered.wires)

        self.amplitude_dimension = (self.n_qubits, 2)
        print("Number of params expected (tapered)", self.amplitude_dimension)

        self.H = H_tapered

        ##################################################################


        ########## Hardware-efficient ansatz (1 layer) ##################
        
        # Define the qnode
        @qml.qnode(dev)
        def circuit(params):

            for q in qubits:
                qml.RY(params[q][0], q)

            for q in qubits[:-1]:
                qml.CNOT((q, q + 1))

            for q in qubits:
                qml.RY(params[q][1], q)

            return qml.expval(self.H)

        self._circuit = circuit

        self.optimized_amplitudes = []

    def simulate(self, amplitudes):
        """Perform the simulation for the molecule.

        Args:
            amplitudes (list): The initial amplitudes (float64).
        Returns:
            float64: The total energy (energy).
        Raise:
            ValueError: If the dimension of the amplitude list is incorrect.
        """

        amplitudes = amplitudes.reshape(self.amplitude_dimension)
        if amplitudes.shape != self.amplitude_dimension:
            raise ValueError("Incorrect dimension for amplitude list.")

        # Optimize the circuit parameters and compute the energy
        energy = self._circuit(amplitudes)
        self.optimized_amplitudes = amplitudes
        print("Energy: ", energy)

        return energy

    def get_rdm(self):
        """Obtain the RDMs from the optimized amplitudes.

        Obtain the RDMs from the optimized amplitudes by using the
        same function for energy evaluation.
        The RDMs are computed by using each fermionic Hamiltonian term,
        transforming them and computing the elements one-by-one.
        Note that the Hamiltonian coefficients will not be multiplied
        as in the energy evaluation.
        The first element of the Hamiltonian is the nuclear repulsion
        energy term, not the Hamiltonian term.

        Returns:
            (numpy.array, numpy.array): One & two-particle RDMs (rdm1_np & rdm2_np, float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """

        if len(self.optimized_amplitudes) == 0:
            raise RuntimeError(
                "Cannot retrieve RDM because no simulation has been run.")


        ############################ NOTE ################################
        
        # Ideally, we should obtain the 1- and 2-RDM matrices using the optimized 
        # variational parameters of the circuit. This makes sense for problem 
        # decomposition methods if these parameters are the ones that minimizes energy.
        # This will work if we do not taper the Hamiltonian. Howerver, if we do taper
        # the Hamiltonian, then it will not work due to the reason mentioned in the
        # tapering section above.

        # If we do not taper the Hamiltonian, then we can use the code belwo the 
        # first return statement. For small molecules, this is feasible but for 
        # larger molecules, the number of terms in the Hamiltonian shoots up and
        # becomes infeasible.

        ##################################################################


        print("Computing RDMs...")

        return self.mean_field.make_rdm1(), self.mean_field.make_rdm2()

        # Save our accurate hamiltonian
        from copy import deepcopy
        orig_hamiltonian = deepcopy(self.f_hamiltonian)
        tmp_H = self.H

        # Initialize the RDM arrays
        rdm1_np = np.zeros((self.of_mole.n_orbitals, ) * 2)
        rdm2_np = np.zeros((self.of_mole.n_orbitals, ) * 4)

        # Loop over each element of Hamiltonian (non-zero value)
        for ikey, key in enumerate(self.f_hamiltonian):

            length = len(key)
            # Treat one-body and two-body term accordingly
            if (length == 2):
                pele, qele = int(key[0][0]), int(key[1][0])
                iele, jele = pele // 2, qele // 2
                # print(pele, qele, iele, jele)
            if (length == 4):
                pele, qele, rele, sele = int(key[0][0]), int(key[1][0]), int(
                    key[2][0]), int(key[3][0])
                iele, jele, kele, lele = pele // 2, qele // 2, rele // 2, sele // 2
                # print(pele, qele, rele, sele, iele, jele, kele, lele)

            # Select the Hamiltonian element (Set coefficient to one)
            # hamiltonian_temp = self.of_mole.get_molecular_hamiltonian()
            hamiltonian_temp = deepcopy(orig_hamiltonian)
            for jkey, key2 in enumerate(hamiltonian_temp):
                hamiltonian_temp[key2] = 1. if (key == key2
                                                and ikey != 0) else 0.

            # Qubitize the element
            qubit_hamiltonian = jordan_wigner(hamiltonian_temp)
            qubit_hamiltonian.compress()

            # Overwrite with the temp hamiltonian
            self.H = qml.qchem.import_operator(qubit_hamiltonian)

            # Calculate the energy with the temp hamiltonian
            opt_energy2 = self.simulate(self.optimized_amplitudes)

            # Put the values in np arrays (differentiate 1- and 2-RDM)
            if (length == 2):
                rdm1_np[iele, jele] = rdm1_np[iele, jele] + opt_energy2
            elif (length == 4):
                if ((iele != lele) or (jele != kele)):
                    rdm2_np[lele, iele, kele,
                            jele] = rdm2_np[lele, iele, kele,
                                            jele] + 0.5 * opt_energy2
                    rdm2_np[iele, lele, jele,
                            kele] = rdm2_np[iele, lele, jele,
                                            kele] + 0.5 * opt_energy2
                else:
                    rdm2_np[iele, lele, jele,
                            kele] = rdm2_np[iele, lele, jele,
                                            kele] + opt_energy2

        # Restore the accurate hamiltonian
        self.H = tmp_H

        return rdm1_np, rdm2_np

    def default_initial_var_parameters(self):
        """ Returns initial variational parameters for a VQE simulation.

        Returns initial variational parameters for the circuit that is generated
        for a given ansatz.

        Returns:
            list: Initial parameters.
        """
        if self.ansatz == self.__class__.Ansatze.UCCSD:
            from dmet.initial_parameters import mp2_initial_amplitudes
            return mp2_initial_amplitudes(self.molecule, self.mean_field)
        else:
            raise RuntimeError(
                "Unsupported ansatz for automatic parameter generation")

    def _build_of_molecule(self, molecule, mean_field):
        """Initialize the instance of Openfermion MolecularData class.

        Interface the pyscf and Openfermion data.
        `pyscf.ao2mo` is used to transform the AO integrals into
        the MO integrals.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.

        Returns:
            openfermion.hamiltonian.MolecularData: Molecular Data in Openfermion (of_mole).
        """
        of_mole = MolecularData(geometry=molecule.atom,
                                basis=molecule.basis,
                                multiplicity=molecule.spin + 1)

        of_mole.mf = mean_field
        of_mole.mol = molecule
        of_mole.n_atoms = molecule.natm
        of_mole.atoms = [row[0] for row in molecule.atom],
        of_mole.protons = 0
        of_mole.nuclear_repulsion = molecule.energy_nuc()
        of_mole.charge = molecule.charge
        of_mole.n_electrons = molecule.nelectron
        of_mole.n_orbitals = len(mean_field.mo_energy)
        of_mole.n_qubits = 2 * of_mole.n_orbitals
        of_mole.hf_energy = mean_field.e_tot
        of_mole.orbital_energies = mean_field.mo_energy
        of_mole.mp2_energy = None
        of_mole.cisd_energy = None
        of_mole.fci_energy = None
        of_mole.ccsd_energy = None
        of_mole.general_calculations = {}
        of_mole._canonical_orbitals = mean_field.mo_coeff
        of_mole._overlap_integrals = mean_field.get_ovlp()
        of_mole.h_core = mean_field.get_hcore()
        of_mole._one_body_integrals = of_mole._canonical_orbitals.T @ of_mole.h_core @ of_mole._canonical_orbitals
        twoint = mean_field._eri
        eri = ao2mo.restore(8, twoint, of_mole.n_orbitals)
        eri = ao2mo.incore.full(eri, of_mole._canonical_orbitals)
        eri = ao2mo.restore(1, eri, of_mole.n_orbitals)
        of_mole._two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1),
                                                 order='C')

        return of_mole
