####
#### Parallel Scan for Linear Recurrences
####
#### Solves: s_t = J_t * s_{t-1} + u_t  for t = 1, ..., T
#### given s_0 (initial state) and transforms (J_t, u_t)
####
#### Uses associative scan with operator:
#### (J₂, u₂) ⊕ (J₁, u₁) = (J₂*J₁, J₂*u₁ + u₂)
####

using LinearAlgebra: mul!, I, Diagonal

####
#### Composition operators for affine transforms
####
#### The key insight: composition of affine transforms is affine.
#### If h₁(x) = J₁*x + u₁ and h₂(x) = J₂*x + u₂, then
#### h₂(h₁(x)) = J₂*(J₁*x + u₁) + u₂ = (J₂*J₁)*x + (J₂*u₁ + u₂)
####

"""
    compose(t2::AbstractAffineTransform, t1::AbstractAffineTransform)

Compose two affine transforms: t2 ∘ t1, meaning t2 is applied after t1.
Returns a new transform representing x ↦ t2(t1(x)).

The composition rule is:
    (J₂, u₂) ⊕ (J₁, u₁) = (J₂*J₁, J₂*u₁ + u₂)
"""
function compose end

# Full matrix composition: (J₂*J₁, J₂*u₁ + u₂)
function compose(t2::MatrixAffineTransform, t1::MatrixAffineTransform)
    J_composed = t2.J * t1.J
    u_composed = t2.J * t1.u + t2.u
    return MatrixAffineTransform(J_composed, u_composed)
end

# Diagonal composition: (d₂ .* d₁, d₂ .* u₁ + u₂)
function compose(t2::DiagonalAffineTransform, t1::DiagonalAffineTransform)
    d_composed = t2.d .* t1.d
    u_composed = t2.d .* t1.u + t2.u
    return DiagonalAffineTransform(d_composed, u_composed)
end

# Block 2x2 composition for leapfrog
# State is [x; v], each dimension d has a 2×2 block:
# [x_d']   [a_d  b_d] [x_d]   [u_x_d]
# [v_d'] = [c_d  e_d] [v_d] + [u_v_d]
#
# Composition: (A₂, u₂) ∘ (A₁, u₁) where A is block-diagonal
# Result block: A₂ * A₁ (2×2 matrix multiply per dimension)
# Result offset: A₂ * u₁ + u₂
function compose(t2::Block2x2AffineTransform, t1::Block2x2AffineTransform)
    # Block matrix multiplication per dimension:
    # [a₂ b₂] [a₁ b₁]   [a₂a₁+b₂c₁  a₂b₁+b₂e₁]
    # [c₂ e₂] [c₁ e₁] = [c₂a₁+e₂c₁  c₂b₁+e₂e₁]
    a_new = t2.a .* t1.a .+ t2.b .* t1.c
    b_new = t2.a .* t1.b .+ t2.b .* t1.e
    c_new = t2.c .* t1.a .+ t2.e .* t1.c
    e_new = t2.c .* t1.b .+ t2.e .* t1.e

    # Offset: A₂ * u₁ + u₂
    # [a₂ b₂] [u₁_x]   [u₂_x]   [a₂*u₁_x + b₂*u₁_v + u₂_x]
    # [c₂ e₂] [u₁_v] + [u₂_v] = [c₂*u₁_x + e₂*u₁_v + u₂_v]
    u_x_new = t2.a .* t1.u_x .+ t2.b .* t1.u_v .+ t2.u_x
    u_v_new = t2.c .* t1.u_x .+ t2.e .* t1.u_v .+ t2.u_v

    return Block2x2AffineTransform(a_new, b_new, c_new, e_new, u_x_new, u_v_new)
end

####
#### Identity transform compositions
####

# Composing with identity from the left: id ∘ t = t
compose(::IdentityMatrixTransform, t::MatrixAffineTransform) = t
compose(::IdentityDiagonalTransform, t::DiagonalAffineTransform) = t
compose(::IdentityBlockTransform, t::Block2x2AffineTransform) = t

# Composing with identity from the right: t ∘ id = t
compose(t::MatrixAffineTransform, ::IdentityMatrixTransform) = t
compose(t::DiagonalAffineTransform, ::IdentityDiagonalTransform) = t
compose(t::Block2x2AffineTransform, ::IdentityBlockTransform) = t

# Two identities
function compose(id1::IdentityMatrixTransform{T}, ::IdentityMatrixTransform) where {T}
    return id1
end
function compose(id1::IdentityDiagonalTransform{T}, ::IdentityDiagonalTransform) where {T}
    return id1
end
function compose(id1::IdentityBlockTransform{T}, ::IdentityBlockTransform) where {T}
    return id1
end

####
#### Apply transform to state
####

"""
    apply(transform::AbstractAffineTransform, x::AbstractVector)

Apply an affine transform to a state vector: x ↦ J*x + u
"""
function apply end

function apply(t::MatrixAffineTransform, x::AbstractVector)
    return t.J * x + t.u
end

function apply(t::DiagonalAffineTransform, x::AbstractVector)
    return t.d .* x + t.u
end

function apply(t::Block2x2AffineTransform, x::AbstractVector)
    D = length(t.a)
    @assert length(x) == 2D "State must be [position; momentum] of length 2D"

    x_pos = @view x[1:D]
    x_mom = @view x[(D + 1):(2D)]

    # Apply block transform per dimension:
    # x'_d = a_d * x_d + b_d * v_d + u_x_d
    # v'_d = c_d * x_d + e_d * v_d + u_v_d
    new_pos = t.a .* x_pos .+ t.b .* x_mom .+ t.u_x
    new_mom = t.c .* x_pos .+ t.e .* x_mom .+ t.u_v

    return vcat(new_pos, new_mom)
end

# Identity transforms
function apply(id::IdentityMatrixTransform{T}, x::AbstractVector) where {T}
    return x
end
function apply(id::IdentityDiagonalTransform{T}, x::AbstractVector) where {T}
    return x
end
function apply(id::IdentityBlockTransform{T}, x::AbstractVector) where {T}
    return x
end

####
#### Parallel Scan Implementation
####

"""
    parallel_scan(transforms::Vector{<:AbstractAffineTransform}, s0::AbstractVector)

Solve the linear recurrence s_t = J_t * s_{t-1} + u_t using parallel scan.

# Arguments
- `transforms`: Vector of affine transforms [(J₁, u₁), (J₂, u₂), ..., (J_T, u_T)]
- `s0`: Initial state s₀

# Returns
- Matrix of shape (T, D) where row t contains s_t

# Algorithm
Uses associative scan with composition operator to compute cumulative transforms
in O(log T) parallel time, then applies each to s0.
"""
function parallel_scan(transforms::Vector{<:AbstractAffineTransform}, s0::AbstractVector)
    T_len = length(transforms)
    D = length(s0)

    # Compute cumulative transforms using associative scan
    # cumulative[t] = transforms[t] ∘ transforms[t-1] ∘ ... ∘ transforms[1]
    #
    # Note: accumulate calls op(accumulator, next_element), but we need
    # compose(next_element, accumulator) to get the right order (next applied after acc)
    cumulative = accumulate((acc, t) -> compose(t, acc), transforms)

    # Apply each cumulative transform to s0 to get the trajectory
    # Note: This is parallelizable (independent for each t)
    trajectory = zeros(eltype(s0), T_len, D)
    for t in 1:T_len
        trajectory[t, :] = apply(cumulative[t], s0)
    end

    return trajectory
end

"""
    parallel_scan!(trajectory::AbstractMatrix, transforms::Vector{<:AbstractAffineTransform}, s0::AbstractVector)

In-place version of parallel_scan. Results are written to `trajectory`.
"""
function parallel_scan!(
    trajectory::AbstractMatrix,
    transforms::Vector{<:AbstractAffineTransform},
    s0::AbstractVector,
)
    T_len = length(transforms)
    D = length(s0)

    @assert size(trajectory) == (T_len, D) "Trajectory must be (T, D) = ($T_len, $D)"

    # Compute cumulative transforms (see parallel_scan for explanation of order)
    cumulative = accumulate((acc, t) -> compose(t, acc), transforms)

    # Apply to s0
    for t in 1:T_len
        trajectory[t, :] = apply(cumulative[t], s0)
    end

    return trajectory
end

####
#### Convenience functions for constructing transforms from Jacobians
####

"""
    make_matrix_transforms(J::Array{T,3}, u::Matrix{T}) where T

Create vector of MatrixAffineTransform from stacked Jacobians and offsets.

# Arguments
- `J`: Array of shape (T, D, D) where J[t, :, :] is the Jacobian at step t
- `u`: Matrix of shape (T, D) where u[t, :] is the offset at step t

# Returns
- Vector of MatrixAffineTransform of length T
"""
function make_matrix_transforms(J::Array{T,3}, u::Matrix{T}) where {T}
    T_len = size(J, 1)
    return [MatrixAffineTransform(J[t, :, :], u[t, :]) for t in 1:T_len]
end

"""
    make_diagonal_transforms(d::Matrix{T}, u::Matrix{T}) where T

Create vector of DiagonalAffineTransform from stacked diagonals and offsets.

# Arguments
- `d`: Matrix of shape (T, D) where d[t, :] is the Jacobian diagonal at step t
- `u`: Matrix of shape (T, D) where u[t, :] is the offset at step t

# Returns
- Vector of DiagonalAffineTransform of length T
"""
function make_diagonal_transforms(d::Matrix{T}, u::Matrix{T}) where {T}
    T_len = size(d, 1)
    return [DiagonalAffineTransform(d[t, :], u[t, :]) for t in 1:T_len]
end

"""
    make_block_transforms(H_diag::Matrix{T}, ε::T, u_x::Matrix{T}, u_v::Matrix{T}) where T

Create vector of Block2x2AffineTransform for leapfrog integration.

# Arguments
- `H_diag`: Matrix of shape (T, D) where H_diag[t, :] is the Hessian diagonal at step t
- `ε`: Step size (scalar)
- `u_x`: Matrix of shape (T, D) for position offsets
- `u_v`: Matrix of shape (T, D) for momentum offsets

# Returns
- Vector of Block2x2AffineTransform of length T
"""
function make_block_transforms(H_diag::Matrix{T}, ε::T, u_x::Matrix{T}, u_v::Matrix{T}) where {T}
    T_len = size(H_diag, 1)
    return [
        Block2x2AffineTransform(
            ones(T, size(H_diag, 2)),              # a = 1
            fill(ε, size(H_diag, 2)),              # b = ε
            ε .* H_diag[t, :],                     # c = ε * H
            ones(T, size(H_diag, 2)) .+ (ε^2) .* H_diag[t, :],  # e = 1 + ε²*H
            u_x[t, :],
            u_v[t, :],
        ) for t in 1:T_len
    ]
end

"""
    make_leapfrog_transforms(H_diag::Matrix{T}, ε::T) where T

Create vector of Block2x2AffineTransform for leapfrog steps with zero offsets.
This is useful for the initial evaluation before computing residuals.

# Arguments
- `H_diag`: Matrix of shape (T, D) where H_diag[t, :] is the Hessian diagonal at step t
- `ε`: Step size (scalar)

# Returns
- Vector of Block2x2AffineTransform of length T (with u_x = u_v = 0)
"""
function make_leapfrog_transforms(H_diag::Matrix{T}, ε::T) where {T}
    T_len, D = size(H_diag)
    u_x = zeros(T, T_len, D)
    u_v = zeros(T, T_len, D)
    return make_block_transforms(H_diag, ε, u_x, u_v)
end

####
#### Sequential scan for comparison/testing
####

"""
    sequential_scan(transforms::Vector{<:AbstractAffineTransform}, s0::AbstractVector)

Solve the linear recurrence sequentially (for testing/comparison).
This is O(T) but serves as a reference implementation.
"""
function sequential_scan(transforms::Vector{<:AbstractAffineTransform}, s0::AbstractVector)
    T_len = length(transforms)
    D = length(s0)

    trajectory = zeros(eltype(s0), T_len, D)
    s_prev = s0

    for t in 1:T_len
        s_t = apply(transforms[t], s_prev)
        trajectory[t, :] = s_t
        s_prev = s_t
    end

    return trajectory
end
