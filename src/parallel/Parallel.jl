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

# Check if we're a submodule of AdvancedHMC
const IS_SUBMODULE =
    parentmodule(@__MODULE__) !== Main && nameof(parentmodule(@__MODULE__)) === :AdvancedHMC

# Import dependencies based on context
if IS_SUBMODULE
    # When loaded as part of AdvancedHMC, use parent's dependencies
    using AbstractMCMC: AbstractMCMC
    using LogDensityProblems: LogDensityProblems
    import ..AdvancedHMC:
        AbstractMetric, DiagEuclideanMetric, UnitEuclideanMetric, DenseEuclideanMetric
    const HAS_ABSTRACTMCMC = true
    const HAS_LOGDENSITYPROBLEMS = true
else
    # Standalone mode: try to load optional dependencies
    const HAS_ABSTRACTMCMC = try
        @eval using AbstractMCMC: AbstractMCMC
        true
    catch
        false
    end
    const HAS_LOGDENSITYPROBLEMS = try
        @eval using LogDensityProblems: LogDensityProblems
        true
    catch
        false
    end
    # Define minimal metric type stubs for standalone testing
    abstract type AbstractMetric end
    struct DiagEuclideanMetric{T} <: AbstractMetric
        M⁻¹::Vector{T}
    end
    struct UnitEuclideanMetric{T,N} <: AbstractMetric
        dim::NTuple{N,Int}
    end
    struct DenseEuclideanMetric{T} <: AbstractMetric
        M⁻¹::Matrix{T}
    end
end

# Types
include("types.jl")

# Parallel scan implementation
include("scan.jl")

# Jacobian computation utilities
include("jacobian.jl")

# Core DEER algorithm
include("deer.jl")

# Parallel MALA
include("mala.jl")

# Parallel HMC
include("hmc.jl")

# AbstractMCMC integration (only if dependencies available)
if HAS_ABSTRACTMCMC && HAS_LOGDENSITYPROBLEMS
    include("abstractmcmc.jl")
end

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

# Export DEER algorithm
export DEERResult
export deer, deer_with_settings, sequential_mcmc

# Export MALA
export MALARandomInputs, MALAConfig
export sample_mala_inputs, mala_proposal, mala_transition
export parallel_mala, sequential_mala

# Export HMC
export HMCRandomInputs, HMCConfig
export sample_hmc_inputs, hmc_transition, hmc_transition_soft
export leapfrog_step, leapfrog_full, hmc_proposal
export parallel_hmc, sequential_hmc
export parallel_leapfrog, leapfrog_transition
export hessian_diagonal_fd

# Export AbstractMCMC integration (only if dependencies available)
if HAS_ABSTRACTMCMC && HAS_LOGDENSITYPROBLEMS
    export AbstractParallelSampler
    export ParallelHMCSampler, ParallelHMC
    export ParallelMALASampler, ParallelMALA
    export ParallelSamplerState, ParallelTransition, ParallelSamplerIterator
    export parallel_sample, get_samples
    export SimpleLogDensity
    # Re-export LogDensityProblems for convenience
    export LogDensityProblems
end

# Export metric types (for standalone testing)
if !IS_SUBMODULE
    export AbstractMetric, DiagEuclideanMetric, UnitEuclideanMetric, DenseEuclideanMetric
end

end # module
