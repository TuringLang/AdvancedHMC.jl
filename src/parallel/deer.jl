####
#### DEER Algorithm: Parallel Newton's Method for MCMC
####
#### Solves the fixed-point problem for MCMC chains:
####   s_t = f_t(s_{t-1})  for t = 1, ..., T
####
#### Using Newton iteration:
####   s_t^{(i+1)} = J_t * s_{t-1}^{(i+1)} + u_t
####   where u_t = f_t(s_{t-1}^{(i)}) - J_t * s_{t-1}^{(i)}
####
#### The linear system is solved in O(log T) time via parallel scan.
####

using Random: AbstractRNG, default_rng

####
#### Core DEER Result Type
####

"""
    DEERResult{T,M}

Result of DEER algorithm execution.

# Fields
- `trajectory::M`: Final state trajectory (T × D)
- `converged::Bool`: Whether the algorithm converged
- `iterations::Int`: Number of Newton iterations performed
- `max_residual::T`: Maximum residual at termination
- `residual_history::Vector{T}`: History of max residuals per iteration
"""
struct DEERResult{T<:AbstractFloat,M<:AbstractMatrix{T}}
    trajectory::M
    converged::Bool
    iterations::Int
    max_residual::T
    residual_history::Vector{T}
end

####
#### Main DEER Algorithm
####

"""
    deer(f, s0, T, ω; method=QuasiDEER(), kwargs...)

Run the DEER algorithm to solve an MCMC chain in parallel.

# Arguments
- `f`: Transition function `f(s_prev, ω_t) -> s_t`
- `s0`: Initial state (vector of length D)
- `T_len`: Number of timesteps (chain length)
- `ω`: Pre-sampled random inputs, accessed as `ω[t]` for timestep t

# Keyword Arguments
- `method::AbstractParallelMethod`: DEER variant to use (default: `QuasiDEER()`)
- `tol::Real`: Convergence tolerance (default: 1e-6)
- `max_iters::Int`: Maximum Newton iterations (default: 1000)
- `jacobian_fn`: Function to compute full Jacobian (default: `jacobian_fd`)
- `jvp_fn`: Function to compute JVP (default: `jvp_fd`)
- `rng::AbstractRNG`: RNG for stochastic methods (default: `default_rng()`)
- `verbose::Bool`: Print convergence info (default: false)

# Returns
- `DEERResult` containing the trajectory and convergence info

# Example
```julia
# Define transition function
f(s, ω) = 0.9 * s + 0.1 * ω

# Pre-sample random inputs
ω = [randn(D) for _ in 1:T]

# Run DEER
result = deer(f, zeros(D), T, ω; method=QuasiDEER())
```
"""
function deer(
    f,
    s0::AbstractVector{T},
    T_len::Int,
    ω;
    method::AbstractParallelMethod=QuasiDEER(),
    tol::Real=1e-6,
    max_iters::Int=1000,
    jacobian_fn=jacobian_fd,
    jvp_fn=jvp_fd,
    rng::AbstractRNG=default_rng(),
    verbose::Bool=false,
) where {T}
    D = length(s0)

    # Initialize trajectory
    trajectory = zeros(T, T_len, D)

    # Initialize with forward pass (optional, can also start with zeros)
    s_prev = s0
    for t in 1:T_len
        trajectory[t, :] = f(s_prev, ω[t])
        s_prev = trajectory[t, :]
    end

    residual_history = T[]

    for iter in 1:max_iters
        # Run one Newton iteration based on method
        trajectory_new = _deer_iteration(
            f, s0, trajectory, ω, method; jacobian_fn=jacobian_fn, jvp_fn=jvp_fn, rng=rng
        )

        # Check convergence
        max_residual = maximum(abs.(trajectory_new - trajectory))
        push!(residual_history, max_residual)

        if verbose && iter % 10 == 0
            println("Iteration $iter: max residual = $max_residual")
        end

        if max_residual < tol
            if verbose
                println("Converged in $iter iterations (residual = $max_residual)")
            end
            return DEERResult(trajectory_new, true, iter, max_residual, residual_history)
        end

        trajectory = trajectory_new
    end

    max_residual = residual_history[end]
    if verbose
        println("Did not converge in $max_iters iterations (residual = $max_residual)")
    end
    return DEERResult(trajectory, false, max_iters, max_residual, residual_history)
end

####
#### Newton Iteration Dispatch
####

function _deer_iteration(f, s0, trajectory, ω, method::FullDEER; jacobian_fn, jvp_fn, rng)
    return _deer_iteration_full(f, s0, trajectory, ω; jacobian_fn=jacobian_fn)
end

function _deer_iteration(f, s0, trajectory, ω, method::QuasiDEER; jacobian_fn, jvp_fn, rng)
    return _deer_iteration_quasi(f, s0, trajectory, ω; jacobian_fn=jacobian_fn)
end

function _deer_iteration(
    f, s0, trajectory, ω, method::StochasticQuasiDEER; jacobian_fn, jvp_fn, rng
)
    return _deer_iteration_stochastic(
        f, s0, trajectory, ω; jvp_fn=jvp_fn, rng=rng, n_samples=method.n_samples
    )
end

####
#### Full DEER Implementation
####

"""
    _deer_iteration_full(f, s0, trajectory, ω; jacobian_fn)

One Newton iteration using full Jacobian matrices.

Memory: O(T * D²)
Work: O(T * D³) for matrix multiplications in scan
"""
function _deer_iteration_full(
    f, s0::AbstractVector{T}, trajectory::AbstractMatrix{T}, ω; jacobian_fn=jacobian_fd
) where {T}
    T_len, D = size(trajectory)

    # Allocate arrays
    f_vals = zeros(T, T_len, D)
    J = zeros(T, T_len, D, D)
    u = zeros(T, T_len, D)

    # Step 1: Evaluate f and compute Jacobians at all timesteps (parallelizable)
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]

        # Evaluate transition function
        f_vals[t, :] = f(s_prev, ω[t])

        # Compute full Jacobian
        f_t(s) = f(s, ω[t])
        J[t, :, :] = jacobian_fn(f_t, s_prev)
    end

    # Step 2: Compute inputs u_t = f_t(s_{t-1}) - J_t * s_{t-1}
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]
        u[t, :] = f_vals[t, :] - J[t, :, :] * s_prev
    end

    # Step 3: Solve linear system via parallel scan
    transforms = make_matrix_transforms(J, u)
    trajectory_new = parallel_scan(transforms, s0)

    return trajectory_new
end

####
#### Quasi-DEER Implementation (Diagonal Jacobian)
####

"""
    _deer_iteration_quasi(f, s0, trajectory, ω; jacobian_fn)

One Newton iteration using diagonal Jacobian approximation.

Memory: O(T * D)
Work: O(T * D) for elementwise operations in scan
"""
function _deer_iteration_quasi(
    f, s0::AbstractVector{T}, trajectory::AbstractMatrix{T}, ω; jacobian_fn=jacobian_fd
) where {T}
    T_len, D = size(trajectory)

    # Allocate arrays
    f_vals = zeros(T, T_len, D)
    J_diag = zeros(T, T_len, D)
    u = zeros(T, T_len, D)

    # Step 1: Evaluate f and compute Jacobian diagonals (parallelizable)
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]

        # Evaluate transition function
        f_vals[t, :] = f(s_prev, ω[t])

        # Compute Jacobian diagonal
        f_t(s) = f(s, ω[t])
        J_diag[t, :] = jacobian_diagonal_full(f_t, s_prev, jacobian_fn)
    end

    # Step 2: Compute inputs u_t = f_t(s_{t-1}) - diag(J_t) .* s_{t-1}
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]
        u[t, :] = f_vals[t, :] - J_diag[t, :] .* s_prev
    end

    # Step 3: Solve linear system via parallel scan (diagonal version)
    transforms = make_diagonal_transforms(J_diag, u)
    trajectory_new = parallel_scan(transforms, s0)

    return trajectory_new
end

####
#### Stochastic Quasi-DEER Implementation (Hutchinson Estimator)
####

"""
    _deer_iteration_stochastic(f, s0, trajectory, ω; jvp_fn, rng, n_samples)

One Newton iteration using stochastic diagonal estimation via Hutchinson's method.

Only requires JVP computations, not full Jacobian.

Memory: O(T * D)
Work: O(T * D * n_samples) for JVP computations
"""
function _deer_iteration_stochastic(
    f,
    s0::AbstractVector{T},
    trajectory::AbstractMatrix{T},
    ω;
    jvp_fn=jvp_fd,
    rng::AbstractRNG=default_rng(),
    n_samples::Int=1,
) where {T}
    T_len, D = size(trajectory)

    # Allocate arrays
    f_vals = zeros(T, T_len, D)
    J_diag = zeros(T, T_len, D)
    u = zeros(T, T_len, D)

    # Step 1: Evaluate f and estimate Jacobian diagonals (parallelizable)
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]

        # Evaluate transition function
        f_vals[t, :] = f(s_prev, ω[t])

        # Estimate Jacobian diagonal via Hutchinson's method
        f_t(s) = f(s, ω[t])
        J_diag[t, :] = hutchinson_diagonal(
            f_t, s_prev, jvp_fn; rng=rng, n_samples=n_samples
        )
    end

    # Step 2: Compute inputs u_t = f_t(s_{t-1}) - diag(J_t) .* s_{t-1}
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]
        u[t, :] = f_vals[t, :] - J_diag[t, :] .* s_prev
    end

    # Step 3: Solve linear system via parallel scan (diagonal version)
    transforms = make_diagonal_transforms(J_diag, u)
    trajectory_new = parallel_scan(transforms, s0)

    return trajectory_new
end

####
#### Block Quasi-DEER for Leapfrog (Phase 4)
####

"""
    _deer_iteration(f, s0, trajectory, ω, method::BlockQuasiDEER; kwargs...)

Dispatch for Block Quasi-DEER method.
"""
function _deer_iteration(
    f, s0, trajectory, ω, method::BlockQuasiDEER; jacobian_fn, jvp_fn, rng
)
    return _deer_iteration_block(
        f,
        s0,
        trajectory,
        ω;
        hessian_diag_fn=method.hessian_diag_fn,
        ε=method.ε,
        M⁻¹=method.M⁻¹,
    )
end

"""
    _deer_iteration_block(f, s0, trajectory, ω; hessian_diag_fn, ε, M⁻¹)

One Newton iteration using 2×2 block-diagonal Jacobian structure for leapfrog.

The state is s = [θ; r] where θ is position and r is momentum.
The Jacobian has 2×2 block structure per dimension:

    J_d = [ 1           ε*M⁻¹_d        ]
          [ ε*H_d       1 + ε²*M⁻¹_d*H_d ]

where H_d is the d-th diagonal element of the Hessian of -log p.

Memory: O(T * D)
Work: O(T * D) for 2×2 block operations in scan
"""
function _deer_iteration_block(
    f,
    s0::AbstractVector{T},
    trajectory::AbstractMatrix{T},
    ω;
    hessian_diag_fn,
    ε::T,
    M⁻¹::AbstractVector{T},
) where {T}
    T_len, state_dim = size(trajectory)
    D = state_dim ÷ 2  # θ and r each have dimension D

    # Allocate arrays
    f_vals = zeros(T, T_len, state_dim)

    # Store block Jacobian components for each timestep
    J_a = zeros(T, T_len, D)  # Top-left diagonal
    J_b = zeros(T, T_len, D)  # Top-right diagonal
    J_c = zeros(T, T_len, D)  # Bottom-left diagonal
    J_e = zeros(T, T_len, D)  # Bottom-right diagonal
    u_x = zeros(T, T_len, D)  # Offset for position
    u_v = zeros(T, T_len, D)  # Offset for momentum

    # Step 1: Evaluate f and compute block Jacobians at all timesteps
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]

        # Evaluate transition function
        f_vals[t, :] = f(s_prev, ω[t])

        # Extract position from previous state
        θ_prev = s_prev[1:D]

        # Compute Hessian diagonal at previous position
        H_diag = hessian_diag_fn(θ_prev)

        # Block Jacobian structure for leapfrog:
        # J = [ I          ε*M⁻¹        ]
        #     [ ε*H_diag   I + ε²*M⁻¹*H_diag ]
        J_a[t, :] .= one(T)
        J_b[t, :] .= ε .* M⁻¹
        J_c[t, :] .= ε .* H_diag
        J_e[t, :] .= one(T) .+ (ε^2) .* M⁻¹ .* H_diag
    end

    # Step 2: Compute offsets u = f(s_prev) - J * s_prev
    for t in 1:T_len
        s_prev = (t == 1) ? s0 : trajectory[t - 1, :]
        θ_prev = s_prev[1:D]
        r_prev = s_prev[(D + 1):end]

        f_θ = f_vals[t, 1:D]
        f_r = f_vals[t, (D + 1):end]

        # u_x = f_θ - (J_a * θ_prev + J_b * r_prev)
        # u_v = f_r - (J_c * θ_prev + J_e * r_prev)
        u_x[t, :] = f_θ .- (J_a[t, :] .* θ_prev .+ J_b[t, :] .* r_prev)
        u_v[t, :] = f_r .- (J_c[t, :] .* θ_prev .+ J_e[t, :] .* r_prev)
    end

    # Step 3: Build block transforms and solve via parallel scan
    transforms = [
        Block2x2AffineTransform(
            J_a[t, :], J_b[t, :], J_c[t, :], J_e[t, :], u_x[t, :], u_v[t, :]
        ) for t in 1:T_len
    ]

    # Initial state split
    θ0 = s0[1:D]
    r0 = s0[(D + 1):end]

    # Run parallel scan
    trajectory_θ, trajectory_r = parallel_scan_block(transforms, θ0, r0)

    # Combine into trajectory
    trajectory_new = zeros(T, T_len, state_dim)
    trajectory_new[:, 1:D] = trajectory_θ
    trajectory_new[:, (D + 1):end] = trajectory_r

    return trajectory_new
end

"""
    parallel_scan_block(transforms, θ0, r0)

Parallel scan for 2×2 block transforms.

Returns (trajectory_θ, trajectory_r) where each is a T_len × D matrix.
"""
function parallel_scan_block(
    transforms::Vector{<:Block2x2AffineTransform{T}},
    θ0::AbstractVector{T},
    r0::AbstractVector{T},
) where {T}
    T_len = length(transforms)
    D = length(θ0)

    # Run prefix sum to get cumulative transforms
    prefix = Vector{Block2x2AffineTransform{T}}(undef, T_len)
    prefix[1] = transforms[1]
    for t in 2:T_len
        prefix[t] = compose(transforms[t], prefix[t - 1])
    end

    # Apply each cumulative transform to initial state
    trajectory_θ = zeros(T, T_len, D)
    trajectory_r = zeros(T, T_len, D)

    for t in 1:T_len
        tr = prefix[t]
        # Apply: [θ'] = [a b] [θ0] + [u_x]
        #        [r']   [c e] [r0]   [u_v]
        trajectory_θ[t, :] = tr.a .* θ0 .+ tr.b .* r0 .+ tr.u_x
        trajectory_r[t, :] = tr.c .* θ0 .+ tr.e .* r0 .+ tr.u_v
    end

    return trajectory_θ, trajectory_r
end

####
#### Utility Functions
####

"""
    deer_with_settings(f, s0, T_len, ω, settings::ParallelMCMCSettings; kwargs...)

Run DEER with settings from a ParallelMCMCSettings struct.
"""
function deer_with_settings(
    f, s0::AbstractVector, T_len::Int, ω, settings::ParallelMCMCSettings; kwargs...
)
    return deer(
        f,
        s0,
        T_len,
        ω;
        method=settings.method,
        tol=settings.tol,
        max_iters=settings.max_iters,
        kwargs...,
    )
end

"""
    sequential_mcmc(f, s0, T_len, ω)

Run MCMC sequentially (for comparison/testing).

Returns the trajectory as a (T × D) matrix.
"""
function sequential_mcmc(f, s0::AbstractVector{T}, T_len::Int, ω) where {T}
    D = length(s0)
    trajectory = zeros(T, T_len, D)

    s_prev = s0
    for t in 1:T_len
        trajectory[t, :] = f(s_prev, ω[t])
        s_prev = trajectory[t, :]
    end

    return trajectory
end
