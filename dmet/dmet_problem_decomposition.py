import warnings

import scipy
from pyscf import scf
from functools import reduce
import pennylane.numpy as np

from dmet.ccsd_solver import CCSDSolver
from dmet.iao_localization import iao_localization

from dmet.dmet_orbitals import dmet_orbitals as _orbitals
from dmet.dmet_fragment import dmet_fragment_constructor as _fragment_constructor
from dmet.dmet_onerdm import dmet_low_rdm as _low_rdm
from dmet.dmet_onerdm import dmet_fragment_rdm as _fragment_rdm
from dmet.dmet_bath import dmet_fragment_bath as _fragment_bath
from dmet.dmet_scf_guess import dmet_fragment_guess as _fragment_guess
from dmet.dmet_scf import dmet_fragment_scf as _fragment_scf


class DMETProblemDecomposition:
    """Employ DMET as a problem decomposition technique.

    DMET single-shot algorithm is used for problem decomposition technique.
    By default, CCSD is used as the electronic structure solver, and
    IAO is used for the localization scheme.
    Users can define other electronic structure solver such as FCI or
    VQE as an impurity solver. Meta-Lowdin localization scheme can be
    used instead of the IAO scheme, which cannot be used for minimal
    basis set.

    Attributes:
        electronic_structure_solver (subclass of ElectronicStructureSolver): A type of electronic structure solver. Default is CCSD.
        electron_localization_method (string): A type of localization scheme. Default is IAO.
    """

    def __init__(self):
        self.verbose = True
        self.electronic_structure_solver = CCSDSolver()
        self.vqe_solver = None
        self.electron_localization_method = iao_localization
        self.frozen = None
        self._act_sp = None
        self._occupied = None
        self._active = None
        self._vqe_idxs = []
        self.newton_max_iter = 1

    def simulate(self, molecule, fragment_atoms, mean_field=None):
        """Perform DMET single-shot calculation.

        If the mean field is not provided it is automatically calculated.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            fragment_atoms (list): List of number of atoms for each fragment (int).
            mean_field (pyscf.scf.RHF): The mean field of the molecule.

        Return:
            float64: The DMET energy (dmet_energy).

        Raise:
            RuntimeError: If the number of atoms in the fragment list agrees with total.
        """

        # Check if the number of fragment sites is equal to the number of atoms in the molecule
        if molecule.natm != sum(fragment_atoms):
            raise RuntimeError(
                "the number of fragment sites is not equal to the number of atoms in the molecule"
            )

        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()

        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn("DMET simulating with mean field not converged.",
                          RuntimeWarning)

        # Construct orbital object
        active_space = range(molecule.nao_nr())

        if self._act_sp is not None:
            active_space = self._act_sp

        print("Active space: ", active_space)
        orbitals = _orbitals(molecule, mean_field, active_space,
                             self.electron_localization_method)

        # TODO: remove last argument, combining fragments not supported
        orb_list, orb_list2, atom_list2 = _fragment_constructor(
            molecule, fragment_atoms)

        print("number of orbitals in each fragment: ", orb_list)
        print("minimum and maximum orbital label in each fragment: ",
              orb_list2)

        # Initialize the energy list and SCF procedure employing newton-raphson algorithm
        energy = []
        chemical_potential = 0.0
        chemical_potential = scipy.optimize.newton(
            self._oneshot_loop,
            chemical_potential,
            args=(orbitals, orb_list, orb_list2, energy),
            tol=1e-10,
            maxiter=self.newton_max_iter,
            disp=False)

        # # Get the final energy value
        niter = len(energy)
        dmet_energy = energy[niter - 1]

        if self.verbose:
            print(' \t*** DMET Cycle Done *** ')
            print(' \tDMET Energy ( a.u. ) = ' +
                  '{:17.10f}'.format(dmet_energy))
            print(' \tChemical Potential   = ' +
                  '{:17.10f}'.format(chemical_potential))

        return dmet_energy

    def _oneshot_loop(self, chemical_potential, orbitals, orb_list, orb_list2,
                      energy_list):
        """Perform the DMET loop.

        This is the function which runs in the minimizer.
        DMET calculation converges when the chemical potential is below the
        threshold value of the Newton-Rhapson optimizer.

        Args:
            chemical_potential (float64): The Chemical potential.
            orbitals (numpy.array): The localized orbitals (float64).
            orb_list (list): The number of orbitals for each fragment (int).
            orb_list2 (list): List of lists of the minimum and maximum orbital label for each fragment (int).
            energy_list (list): List of DMET energy for each iteration (float64).

        Returns:
            float64: The new chemical potential.
        """

        # Calculate the 1-RDM for the entire molecule
        onerdm_low = _low_rdm(orbitals.active_fock,
                              orbitals.number_active_electrons)

        niter = len(energy_list) + 1

        if self.verbose:
            print(" \tIteration = ", niter)
            print(' \t----------------')
            print(' ')

        number_of_electron = 0.0
        energy_temp = 0.0

        if (self._occupied is not None) and (self._active is not None):
            assert len(self._occupied) == len(self._active)
            assert len(self._occupied) == len(orb_list)
            occupied = self._occupied
            active = self._active
        else:
            occupied = [None] * len(orb_list)
            active = [None] * len(orb_list)

        for i, norb in enumerate(orb_list):
            if self.verbose:
                print("\t\tFragment Number : # ", i + 1)
                print('\t\t------------------------')

            t_list = []
            t_list.append(norb)
            temp_list = orb_list2[i]
            print("\t\tt list: ", t_list)
            print()

            # Construct bath orbitals
            bath_orb, e_occupied = _fragment_bath(orbitals.mol_full, t_list,
                                                  temp_list, onerdm_low)
            print('\t\tBath Orbitals Constructed', bath_orb.shape)
            print('\t\te_occupied', e_occupied.shape)
            print("\t\tt list after fragmet bath: ", t_list)
            print()

            # Obtain one particle rdm for a fragment
            norb_high, nelec_high, onerdm_high = _fragment_rdm(
                t_list, bath_orb, e_occupied, orbitals.number_active_electrons)
            print('\t\tOne Particle RDM Constructed', onerdm_high.shape)
            print('\t\tNumber of Electrons in Fragment', nelec_high)
            print('\t\tNumber of Orbitals in Fragment', norb_high)
            print()

            # Obtain one particle rdm for a fragment
            one_ele, fock, two_ele = orbitals.dmet_fragment_hamiltonian(
                bath_orb, norb_high, onerdm_high)
            print('\t\tOne Particle Hamiltonian Constructed', one_ele.shape)
            print('\t\tFock Matrix Constructed', fock.shape)
            print('\t\tTwo Particle Hamiltonian Constructed', two_ele.shape)
            print()

            # Construct guess orbitals for fragment SCF calculations
            guess_orbitals = _fragment_guess(t_list, bath_orb,
                                             chemical_potential, norb_high,
                                             nelec_high, orbitals.active_fock)
            print('\t\tGuess Orbitals Constructed', guess_orbitals.shape)
            print()

            # Carry out SCF calculation for a fragment
            mf_fragment, fock_frag_copy, mol_frag = _fragment_scf(
                t_list, two_ele, fock, nelec_high, norb_high, guess_orbitals,
                chemical_potential)
            print('\t\tSCF Calculation Done', mf_fragment.e_tot)
            print('\t\tFock Matrix Constructed', fock_frag_copy.shape)
            print('\t\tMolecule fragment Constructed', mol_frag)
            print()

            # Solve the electronic structure
            if i in self._vqe_idxs:
                energy = self.vqe_solver.simulate(mol_frag,
                                                  mf_fragment,
                                                  occupied=occupied[i],
                                                  active=active[i],
                                                  frozen=self.frozen)
            else:
                energy = self.electronic_structure_solver.simulate(
                    mol_frag,
                    mf_fragment,
                    occupied=occupied[i],
                    active=active[i],
                    frozen=self.frozen)
            print('\t\tElectronic Structure Solver Done', energy)
            print()

            # Calculate the RDMs
            if i in self._vqe_idxs:
                cc_onerdm, cc_twordm = self.vqe_solver.get_rdm()
            else:
                cc_onerdm, cc_twordm = self.electronic_structure_solver.get_rdm(
                )
            print('\t\tCC One RDM Constructed', cc_onerdm.shape)
            print('\t\tCC Two RDM Constructed', cc_twordm.shape)
            print()

            # Compute the fragment energy
            fragment_energy, total_energy_rdm, one_rdm = self._compute_energy(
                mf_fragment, cc_onerdm, cc_twordm, fock_frag_copy, t_list,
                one_ele, two_ele, fock)
            print('\t\tFragment Energy Computed', fragment_energy)
            print('\t\tTotal Energy RDM Computed', total_energy_rdm)
            print('\t\tOne RDM Computed', one_rdm.shape)
            print()

            # Sum up the energy
            energy_temp += fragment_energy

            # Sum up the number of electrons
            number_of_electron += np.trace(one_rdm[:t_list[0], :t_list[0]])

            if self.verbose:
                print("\t\tFragment Energy                 = " +
                      '{:17.10f}'.format(fragment_energy))
                print("\t\tNumber of Electrons in Fragment = " +
                      '{:17.10f}'.format(np.trace(one_rdm)))
                print()

        energy_temp += orbitals.core_constant_energy
        energy_list.append(energy_temp)

        print(f"DMET energy at iteration {niter} is {energy_temp} Ha")
        print(
            f"Number of electrons at iteration {niter} is {number_of_electron}"
        )
        print(
            f"Number of active electrons is {orbitals.number_active_electrons}"
        )
        print()

        return number_of_electron - orbitals.number_active_electrons

    def _compute_energy(self, mf_frag, onerdm, twordm, fock_frag_copy, t_list,
                        oneint, twoint, fock):
        """Calculate the fragment energy.

        Args:
            mean_field (pyscf.scf.RHF): The mean field of the fragment.
            cc_onerdm (numpy.array): one-particle reduced density matrix (float64).
            cc_twordm (numpy.array): two-particle reduced density matrix (float64).
            fock_frag_copy (numpy.array): Fock matrix with the chemical potential subtracted (float64).
            t_list (list): List of number of fragment and bath orbitals (int).
            oneint (numpy.array): One-electron integrals of fragment (float64).
            twoint (numpy.array): Two-electron integrals of fragment (float64).
            fock (numpy.array): Fock matrix of fragment (float64).

        Returns:
            float64: Fragment energy (fragment_energy).
            float64: Total energy for fragment using RDMs (total_energy_rdm).
            numpy.array: One-particle RDM for a fragment (one_rdm, float64).
        """

        # Execute CCSD calculation
        norb = t_list[0]

        # Calculate the one- and two- RDM for DMET energy calculation (Transform to AO basis)
        one_rdm = reduce(np.dot,
                         (mf_frag.mo_coeff, onerdm, mf_frag.mo_coeff.T))

        twordm = np.einsum('pi,ijkl->pjkl', mf_frag.mo_coeff, twordm)
        twordm = np.einsum('qj,pjkl->pqkl', mf_frag.mo_coeff, twordm)
        twordm = np.einsum('rk,pqkl->pqrl', mf_frag.mo_coeff, twordm)
        twordm = np.einsum('sl,pqrl->pqrs', mf_frag.mo_coeff, twordm)

        # Calculate the total energy based on RDMs
        total_energy_rdm = np.einsum(
            'ij,ij->', fock_frag_copy,
            one_rdm) + 0.5 * np.einsum('ijkl,ijkl->', twoint, twordm)

        # Calculate fragment expectation value
        fragment_energy_one_rdm = 0.25  * np.einsum('ij,ij->',     one_rdm[ : norb, :      ], fock[: norb, :     ] + oneint[ : norb, :     ]) \
                               + 0.25  * np.einsum('ij,ij->',     one_rdm[ :     , : norb ], fock[:     , : norb] + oneint[ :     , : norb])

        fragment_energy_twordm = 0.125 * np.einsum('ijkl,ijkl->', twordm[ : norb, :     , :     , :      ], twoint[ : norb, :     , :     , :     ]) \
                               + 0.125 * np.einsum('ijkl,ijkl->', twordm[ :     , : norb, :     , :      ], twoint[ :     , : norb, :     , :     ]) \
                               + 0.125 * np.einsum('ijkl,ijkl->', twordm[ :     , :     , : norb, :      ], twoint[ :     , :     , : norb, :     ]) \
                               + 0.125 * np.einsum('ijkl,ijkl->', twordm[ :     , :     , :     , : norb ], twoint[ :     , :     , :     , : norb])

        fragment_energy = fragment_energy_one_rdm + fragment_energy_twordm

        return fragment_energy, total_energy_rdm, one_rdm
