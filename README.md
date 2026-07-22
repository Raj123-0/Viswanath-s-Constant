===============================================================================
PROJECT: Viswanath's Constant Computation Engine
===============================================================================

OVERVIEW:
Calculates Viswanath's Constant (K ≈ 1.13198824...) to N significant digits. 
Viswanath's constant describes the exponential growth rate of a random 
Fibonacci sequence, where terms are added or subtracted with equal probability 1/2.

ALGORITHM & IMPLEMENTATION:
- Flattened DFS with Memoization: Evaluates logarithmic sum of vector norms over 
  a random tree DAG using state-vector reduction (f_n, f_{n-1}).
- Algebraic Symmetry Pruning: Canonicalizes state vectors (forcing u >= 0) to 
  prune duplicate matrix trajectories.
- Multi-Core Work Distribution: Parallelizes subtree traversals across worker processes.
