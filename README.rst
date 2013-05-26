SudoCUDA
========

Some experiments with CUDA and Monte Carlo Sudoku Solvers.

The code in this repository is the result of my efforts to learn CUDA, and is not a very good, fast or efficient way to solve sudokus.

The algorithm is a simple Metropolis, employing a Hamiltonian with a unique ground state corresponding to the solution. I chose it because it can be parallelized very easily.
