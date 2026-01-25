####
#### Types for Parallel MCMC (DEER algorithm)
####

"""
    AbstractParallelMethod

Abstract type for parallel MCMC methods based on the DEER algorithm.
"""
abstract type AbstractParallelMethod end

"""
    FullDEER <: AbstractParallelMethod

Full DEER method using complete Jacobian matrices.
Best for low-dimensional problems where O(T * D²) memory is acceptable.
"""
struct FullDEER <: AbstractParallelMethod end

"""
    QuasiDEER <: AbstractParallelMethod

Quasi-DEER method using diagonal Jacobian approximation.
Memory efficient: O(T * D) instead of O(T * D²).
"""
struct QuasiDEER <: AbstractParallelMethod end

"""
    StochasticQuasiDEER{N} <: AbstractParallelMethod

Stochastic Quasi-DEER using Hutchinson's method to estimate Jacobian diagonal.
Only requires JVP (Jacobian-vector product) instead of full Jacobian computation.

# Fields
- `n_samples::N`: Number of Monte Carlo samples for diagonal estimation (default: 1)
"""
struct StochasticQuasiDEER{N} <: AbstractParallelMethod
    n_samples::N
end
StochasticQuasiDEER() = StochasticQuasiDEER(1)

"""
    BlockQuasiDEER{T,V,F} <: AbstractParallelMethod

Block Quasi-DEER for leapfrog integration.
Keeps 2×2 block structure per dimension to capture position-momentum interactions.
Converges ~2× faster than diagonal Quasi-DEER for HMC leapfrog.

# Fields
- `hessian_diag_fn::F`: Function to compute diagonal of Hessian of -log p
- `ε::T`: Leapfrog step size
- `M⁻¹::V`: Inverse mass matrix diagonal
"""
struct BlockQuasiDEER{T<:AbstractFloat,V<:AbstractVector{T},F} <: AbstractParallelMethod
    hessian_diag_fn::F
    ε::T
    M⁻¹::V
end

"""
    BlockQuasiDEER(hessian_diag_fn, ε, D)

Construct BlockQuasiDEER with unit mass matrix.
"""
function BlockQuasiDEER(hessian_diag_fn, ε::T, D::Int) where {T}
    return BlockQuasiDEER(hessian_diag_fn, ε, ones(T, D))
end

####
#### Affine Transform Types for Parallel Scan
####

"""
    AbstractAffineTransform{T}

Abstract type for affine transforms used in parallel scan.
An affine transform represents: x ↦ J * x + u
"""
abstract type AbstractAffineTransform{T} end

"""
    MatrixAffineTransform{T,M,V} <: AbstractAffineTransform{T}

Full matrix affine transform: x ↦ J * x + u where J is a D×D matrix.
Used in full DEER algorithm.

# Fields
- `J::M`: Jacobian matrix (D × D)
- `u::V`: Offset vector (D,)
"""
struct MatrixAffineTransform{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractAffineTransform{T}
    J::M
    u::V
end

"""
    DiagonalAffineTransform{T,V} <: AbstractAffineTransform{T}

Diagonal affine transform: x ↦ diag(d) * x + u where d is a vector.
Used in Quasi-DEER algorithm. Much more memory efficient than full matrix.

# Fields
- `d::V`: Diagonal of Jacobian (D,)
- `u::V`: Offset vector (D,)
"""
struct DiagonalAffineTransform{T,V<:AbstractVector{T}} <: AbstractAffineTransform{T}
    d::V
    u::V
end

"""
    Block2x2AffineTransform{T,V} <: AbstractAffineTransform{T}

Block-diagonal affine transform for leapfrog integration.
State is [x; v] (position and momentum, concatenated).

For each dimension d, the 2×2 block is:
```
[x_d']   [a_d  b_d] [x_d]   [u_x_d]
[v_d'] = [c_d  e_d] [v_d] + [u_v_d]
```

For a single leapfrog step with Hessian diagonal h_d:
- a_d = 1, b_d = ε, c_d = ε*h_d, e_d = 1 + ε²*h_d

After composition, the blocks can have arbitrary values, so we store all four.

# Fields
- `a::V`: Top-left diagonal (D,)
- `b::V`: Top-right diagonal (D,)
- `c::V`: Bottom-left diagonal (D,)
- `e::V`: Bottom-right diagonal (D,)
- `u_x::V`: Offset for position (D,)
- `u_v::V`: Offset for momentum (D,)
"""
struct Block2x2AffineTransform{T,V<:AbstractVector{T}} <: AbstractAffineTransform{T}
    a::V
    b::V
    c::V
    e::V
    u_x::V
    u_v::V
end

"""
    Block2x2AffineTransform(H_diag::AbstractVector, ε::Real)

Construct a Block2x2AffineTransform for a single leapfrog step.

For leapfrog with Hessian diagonal H_diag and step size ε:
- a = 1, b = ε, c = ε*H_diag, e = 1 + ε²*H_diag
- u_x = u_v = 0
"""
function Block2x2AffineTransform(H_diag::AbstractVector{T}, ε::T) where {T}
    D = length(H_diag)
    a = ones(T, D)
    b = fill(ε, D)
    c = ε .* H_diag
    e = ones(T, D) .+ (ε^2) .* H_diag
    u_x = zeros(T, D)
    u_v = zeros(T, D)
    return Block2x2AffineTransform(a, b, c, e, u_x, u_v)
end

####
#### Identity transforms (for initial state)
####

"""
    IdentityMatrixTransform{T,N} <: AbstractAffineTransform{T}

Identity transform for full matrix case: x ↦ I * x + 0

# Fields
- `dim::N`: Dimension of the state space
"""
struct IdentityMatrixTransform{T,N} <: AbstractAffineTransform{T}
    dim::N
end

"""
    IdentityDiagonalTransform{T,N} <: AbstractAffineTransform{T}

Identity transform for diagonal case: x ↦ 1 .* x + 0

# Fields
- `dim::N`: Dimension of the state space
"""
struct IdentityDiagonalTransform{T,N} <: AbstractAffineTransform{T}
    dim::N
end

"""
    IdentityBlockTransform{T,N} <: AbstractAffineTransform{T}

Identity transform for block-diagonal case.

# Fields
- `dim::N`: Dimension D (not 2D) of the position/momentum space
"""
struct IdentityBlockTransform{T,N} <: AbstractAffineTransform{T}
    dim::N
end

####
#### Parallel MCMC Settings
####

"""
    ParallelMCMCSettings{M,T}

Settings for parallel MCMC algorithms.

# Fields
- `method::M`: Parallel method to use (FullDEER, QuasiDEER, etc.)
- `tol::T`: Convergence tolerance for Newton iterations
- `max_iters::Int`: Maximum number of Newton iterations
- `window_size::Union{Int,Nothing}`: Window size for sliding window (nothing = full sequence)
"""
struct ParallelMCMCSettings{M<:AbstractParallelMethod,T<:AbstractFloat}
    method::M
    tol::T
    max_iters::Int
    window_size::Union{Int,Nothing}
end

function ParallelMCMCSettings(;
    method::AbstractParallelMethod=QuasiDEER(),
    tol::AbstractFloat=1e-6,
    max_iters::Int=1000,
    window_size::Union{Int,Nothing}=nothing,
)
    T = typeof(tol)
    return ParallelMCMCSettings{typeof(method),T}(method, tol, max_iters, window_size)
end

####
#### Parallel MCMC State
####

"""
    ParallelMCMCState{T,M}

State of the parallel MCMC algorithm during Newton iterations.

# Fields
- `trajectory::M`: Current state trajectory (T × D)
- `n_iters::Int`: Number of Newton iterations performed
- `converged::Bool`: Whether the algorithm has converged
- `max_residual::T`: Maximum residual at convergence (or last iteration)
"""
mutable struct ParallelMCMCState{T<:AbstractFloat,M<:AbstractMatrix{T}}
    trajectory::M
    n_iters::Int
    converged::Bool
    max_residual::T
end

function ParallelMCMCState(trajectory::AbstractMatrix{T}) where {T}
    return ParallelMCMCState{T,typeof(trajectory)}(trajectory, 0, false, T(Inf))
end
