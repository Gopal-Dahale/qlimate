"""Perform fragment SCF calculation.

The fragment SCF calculation for DMET calculation
is done here.

"""

import time
from pyscf import gto, scf, ao2mo
from functools import reduce
import numpy as np


def dmet_fragment_scf(t_list, two_ele, fock, number_electrons, number_orbitals,
                      guess_orbitals, chemical_potential):
    """Perform SCF calculation.

    Args:
        t_list (list): Number of [0] fragment & [1] bath orbitals (int).
        two_ele (numpy.array): Two-electron integrals for fragment calculation (float64).
        fock (numpy.array): The fock matrix for fragment calculation (float64).
        number_electrons (int): Number of electrons for fragment calculation.
        number_orbitals (int): Number of orbitals for fragment calculation.
        guess_orbitals (numpy.array): Guess orbitals for SCF calculation (float64).
        chemical_potential (float64): The chemical potential.

    Returns:
        mf_frag (pyscf.scf.RHF): The mean field of the molecule (Fragment calculation).
        fock_frag_copy (numpy.array): The fock matrix with chemical potential subtracted (float64).
        mol_frag (pyscf.gto.Mole): The molecule to simulate (Fragment calculation).
    """

    # Deep copy the fock matrix
    fock_frag_copy = fock.copy()

    # Subtract the chemical potential to make the number of electrons consistent
    if (chemical_potential != 0.0):
        for orb in range(t_list[0]):
            fock_frag_copy[orb, orb] -= chemical_potential

    # Determine the molecular space (set molecule object of pyscf)
    mol_frag = gto.Mole()
    mol_frag.build(verbose=0)
    mol_frag.atom.append(('C', (0, 0, 0)))
    mol_frag.nelectron = number_electrons
    mol_frag.incore_anyway = True

    # Perform SCF calculation (set mean field object of pyscf)
    start = time.time()
    mf_frag = scf.RHF(mol_frag)
    print("Time for SCF RHF object: ", time.time() - start)

    mf_frag.get_hcore = lambda *args: fock_frag_copy
    mf_frag.get_ovlp = lambda *args: np.eye(number_orbitals)

    start = time.time()
    mf_frag._eri = ao2mo.restore(8, two_ele, number_orbitals)
    print("Time for ao2mo.restore: ", time.time() - start)

    start = time.time()
    mf_frag.scf(guess_orbitals)
    print("Time for SCF: ", time.time() - start)

    # Calculate the density matrix for the fragment
    dm_frag = reduce(
        np.dot,
        (mf_frag.mo_coeff, np.diag(mf_frag.mo_occ), mf_frag.mo_coeff.T))

    # Use newton-raphson algorithm if the above SCF calculation is not converged
    if (mf_frag.converged == False):
        mf_frag.get_hcore = lambda *args: fock_frag_copy
        mf_frag.get_ovlp = lambda *args: np.eye(number_orbitals)
        mf_frag._eri = ao2mo.restore(8, two_ele, number_orbitals)
        mf_frag = scf.RHF(mol_frag).newton()
        energy = mf_frag.kernel(dm0=dm_frag)
        dm_frag = reduce(
            np.dot,
            (mf_frag.mo_coeff, np.diag(mf_frag.mo_occ), mf_frag.mo_coeff.T))

    return mf_frag, fock_frag_copy, mol_frag
