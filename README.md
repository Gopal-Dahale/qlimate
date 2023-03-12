# Qlimate: Deloitte’s Quantum Climate Challenge 2023

## Team Jetix

Members: Gopal Ramesh Dahale

## Abstract
One of the frontiers for the Noisy Intermediate-Scale Quantum (NISQ) era's practical uses of quantum computers is quantum chemistry. By employing Hybrid Quantum Classical Optimisation we aim to investigate the minimum of the Ground Potential Energy Surface (PES) of the MOF with gas molecules. We used a deparameterisation approach [1] to freeze $R_y$ gates with standardized parameter values which helped in simplifying the energy landscape while maintaining the accuracy of the global minimum.

We extended the deparameterisation procedure to carbon capture on MOFs and explored the minimum of PES with $CO_2$ with 2 metal ions $Mn(II)$ and $Cu(I)$. Also $H_2O$ with $Mn(II)$. For both the systems, we were able to reduce the parameters to 2 from 8 and 10 for the ansatz of $Mn(II)$ and $Cu(I)$ respectively. For $H_2O + Mn(II)$ using active space reduction, we simulated the ansatz with 4 qubits and achieved a relative error of $10^{-2}$ Ha with ideal simulation and $10^{-1}$ Ha with noisy simulation. We employed error mitigation techniques in noisy simulators which converged the ground state energy to within 6 percent of the actual.

For capturing $CO_2$, amine scrubbing is a promising technology that is nearly ready to be applied industrially. We modelled $CO_2 + CH_3NH_2$ system and calculated the PES. We simulated the ansatz with 7 qubits and achieved a relative error of  $10^{−2}$ Ha with ideal simulation and  $10^{−1}$ Ha with noisy simulation. With Braket's DM1 Simulator we obtained a relative error within $10^{−2}$ Ha. The ground state energy was within 3.6 percent of the actual. The $CO_2 + CH_3NH_2$ system was then modelled using Density Matrix Embedding Theory (DMET) [2] [3] [4], first using classical CCSD solver for every fragment and then using VQE as fragment solver for Nitrogen. With some approximations, the simulation results are within $10^{-1}$ Ha. 

Finally, we explored $H_2O + Cu-MOF-74$ system using DMET and a simple fragmentation strategy. We were able to run the algorithm in feasible time and compared the results with RHF method. The results are promising and within $10^{-2}$ Ha relative error. In future, we can extend the method to use quantum computing method for high accuracy solver.

## Files

1. The first is the `qhack-simulation.ipynb` which describes the problem statement and simulates the ansatzes. 
2. Based on the deparameterised circuits obtained in the previous step, `qhack-hardware.ipynb` executes them on real hardware.
3. Although, the notebooks are self contained, `report.pdf` summarizes all the problem statement, methodology, results and conclusion.

## References
1. [Molecular Energy Landscapes of Hardware-Efficient Ansätze in Quantum Computing Boy Choy* and David J. Wales](https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.2c01057)
2. [Density Matrix Embedding: A Strong-Coupling Quantum Embedding Theory Gerald Knizia* and Garnet Kin-Lic Chan](https://pubs.acs.org/doi/10.1021/ct301044e)
3. [A Practical Guide to Density Matrix Embedding Theory in Quantum Chemistry Sebastian Wouters*, Carlos A. Jiménez-Hoyos, Qiming Sun, and Garnet K.-L. Chan*](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00316)
4. [OpenQEMIST-DMET](http://openqemist.1qbit.com/docs/dmet_microsoft_qsharp.html)
