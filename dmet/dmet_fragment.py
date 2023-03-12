"""Construct the orbital list for the fragments.

Construct the orbital list, showing how much orbitals
to be included for each fragment, from the atom list, 
showing how many atoms included in each fragment.

"""

def dmet_fragment_constructor(mol, atom_list):
    """Construct orbital list.

    Make a list of number of orbitals for each fragment
    while obtaining the list if we consider combining
    fragments.

    Args: 
        mol (pyscf.gto.Mole): The molecule to simulate (The full molecule).
        atom_list (list): The atom list for each fragment (int).
        
    Returns:
        orb_list (list): The number of orbitals for each fragment (int).
        orb_list2 (list): List of lists of the minimum and maximum orbital label for each fragment (int).
        atom_list2 (list): The new atom list for each fragment (int).
    """

    # Make a new atom list based on how many fragments for DMET calculation
    atom_list2 = atom_list
   

    # Initialize the list of the number of orbitals
    orb_list = []
    orb_list2 = []
    isum = 0
    isum2 = -1
    iorb = 0
    jorb = 0

    # Calculate the number of orbitals for each atom
    for i in atom_list2:
        itemp = 0
        isum2 += i
        for total_basis in mol.spheric_labels():
            print(total_basis)
            item = total_basis.split()
            item0 = int(item[0])
            if ((item0 >= isum) and (item0 <= isum2)):
                itemp += 1
        isum += i
        jorb += itemp
        orb_list.append(itemp)
        orb_list2.append([iorb, jorb])
        iorb += itemp

    return orb_list, orb_list2, atom_list2
