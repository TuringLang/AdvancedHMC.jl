"""
    Parallel

Submodule for parallel MCMC algorithms based on the DEER (Doubly Efficient Estimation
via Recursion) algorithm from Zoltowski et al., "Parallelizing MCMC Across the
Sequence Length" (NeurIPS 2025).

The key insight is that MCMC chains can be viewed as nonlinear recursions that can be
solved via Newton's method. Each Newton iteration involves solving a linear system
that can be computed in O(log T) time using parallel scan.

# Main Components

- **Parallel Scan**: Core algorithm for solving linear recurrences in O(log T) time
- **Affine Transforms**: Types representing the Jacobian and offset at each step
- **DEER Methods**: Full DEER, Quasi-DEER, Stochastic Quasi-DEER, Block Quasi-DEER

# Usage

```julia
using AdvancedHMC.Parallel

# Create diagonal affine transforms (for Quasi-DEER)
d = rand(100, 10)  # T=100 steps, D=10 dimensions
u = rand(100, 10)  # offsets
transforms = make_diagonal_transforms(d, u)

# Solve linear recurrence s_t = d_t .* s_{t-1} + u_t
s0 = zeros(10)
trajectory = parallel_scan(transforms, s0)
```
"""
module Parallel

using LinearAlgebra
using Random

# Types
include("types.jl")

# Parallel scan implementation
include("scan.jl")

# Jacobian computation utilities
include("jacobian.jl")

# Export types
export AbstractParallelMethod, FullDEER, QuasiDEER, StochasticQuasiDEER, BlockQuasiDEER

export AbstractAffineTransform,
    MatrixAffineTransform,
    DiagonalAffineTransform,
    Block2x2AffineTransform,
    IdentityMatrixTransform,
    IdentityDiagonalTransform,
    IdentityBlockTransform

export ParallelMCMCSettings, ParallelMCMCState

# Export scan functions
export compose, apply
export parallel_scan, parallel_scan!, sequential_scan
export make_matrix_transforms, make_diagonal_transforms
export make_block_transforms, make_leapfrog_transforms

# Export Jacobian utilities
export jacobian_fd, jacobian_diagonal_full, batch_jacobians, batch_jacobian_diagonals
export jvp_fd, vjp_fd
export rademacher_vector, hutchinson_diagonal, batch_hutchinson_diagonals
export hessian_diagonal, batch_hessian_diagonals

end # module
