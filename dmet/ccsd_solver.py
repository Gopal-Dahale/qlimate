"""Perform CCSD calculation.

The electronic structure calculation employing the 
coupled-cluster singles and doubles (CCSD) method 
is done here.

"""

import time
import warnings
from pyscf import cc, scf


class CCSDSolver:
    """Perform CCSD calculation.
    
    Uses the CCSD method to solve the electronic structure problem.
    PySCF program will be utilized. 
    Users can also provide a function that takes a `pyscf.gto.Mole` 
    as its first argument and `pyscf.scf.RHF` as its second.

    Attributes:
        cc_fragment (pyscf.cc.CCSD): The coupled-cluster object.
    """

    def __init__(self):
        self.cc_fragment = None

    def simulate(self, molecule, mean_field=None, **kwargs):
        """Perform the simulation (energy calculation) for the molecule.

        If the mean field is not provided it is automatically calculated.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.

        Returns:
            float64: CCSD energy (total_energy).
        """
        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()

        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn(
                "CCSDSolver simulating with mean field not converged.",
                RuntimeWarning)

        print("Number of orbitals: ", mean_field.mo_occ.shape[0])
        print("Frozen orbitals: ", kwargs.get('frozen', 0))

        # Execute CCSD calculation

        start = time.time()
        self.cc_fragment = cc.ccsd.CCSD(mean_field,
                                        frozen=kwargs.get('frozen', 0))
        self.cc_fragment.verbose = 0
        self.cc_fragment.conv_tol = 1e-9
        self.cc_fragment.conv_tol_normt = 1e-7
        correlation_energy, t1, t2 = self.cc_fragment.ccsd()
        print("CCSD time: ", time.time() - start)

        scf_energy = mean_field.e_tot
        total_energy = scf_energy + correlation_energy

        return total_energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle RDMs.

        Calculate the CCSD reduced density matrices. 
        The CCSD lambda equation will be solved for calculating 
        the RDMs. 

        Returns:
            (numpy.array, numpy.array): One & two-particle RDMs (cc_onerdm & cc_twordm, float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """

        # Check if CCSD calculation is performed
        if not self.cc_fragment:
            raise RuntimeError(
                "Cannot retrieve RDM because no simulation has been run.")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        start = time.time()
        self.cc_fragment.solve_lambda()
        print("Lambda time: ", time.time() - start)

        start = time.time()
        cc_onerdm = self.cc_fragment.make_rdm1()
        print("1-RDM time: ", time.time() - start)

        start = time.time()
        cc_twordm = self.cc_fragment.make_rdm2()
        print("2-RDM time: ", time.time() - start)

        print(cc_onerdm.shape, cc_twordm.shape)

        return cc_onerdm, cc_twordm
