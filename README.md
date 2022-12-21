This repository contains the code for the paper

# Deep Quadratic Hedging

by A. Gnoatto, S. Lavagnini and A. Picarelli.

The code is divided into four different folders:
1) DeepQuadraticHedging_MeanVariance contains the code for deep mean-variance hedging in dimension m = 1;
2) DeepQuadraticHedging_LocalRisk contains the code for deep local risk minimization in dimension m = 1;
3) MultiDeepQuadraticHedging_MeanVariance contains the code for deep mean-variance hedging in dimension m > 1;
4) MultiDeepQuadraticHedging_LocalRisk contains the code for deep local risk minimization in dimension m > 1.

Each folder contains:
i) mainScriptHeston.py with the main code to be run for simulations/training/pricing;
ii) HestonEquation.py with the code for defining the Heston model and the related functions for the (first) BSDE;
iii) solver.py with the code for the deep BDSE solver;
iv) equation.py with an auxiliary class.

In addition, DeepQuadraticHedging_MeanVariance and DeepQuadraticHedging_LocalRisk contain:
v) HestonPDEsolver.py with the code for solving PDEs used for benchmarking the solver solutions in the local risk case;
vi) HestonPDEsolver_2.py with the code for solving PDEs used for benchmarking the solver solutions in the mean-variance case 
(where there is a time-dependent coefficient);

Moreover, DeepQuadraticHedging_MeanVariance and MultiDeepQuadraticHedging_MeanVariance contain:
vii) RecursiveEquation.py with the code for defining the Heston model and the related functions for the second BSDE;
viii) RecursiveSolver.py with the code for the deep BSDE solver used for the second BSDE;

Finally, each folder contains subfolder(s) where the models corresonding to N = 10 discretization points have been saved.
These can be loaded, so to avoid training the solver.
