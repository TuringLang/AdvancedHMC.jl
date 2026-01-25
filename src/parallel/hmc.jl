####
#### Parallel HMC Implementation
####
#### Two approaches for parallelizing HMC:
#### - Approach A: Parallelize across HMC steps (T samples)
#### - Approach B: Parallelize leapfrog steps within each HMC sample
####

using Random: AbstractRNG, default_rng

####
#### HMC Random Inputs
####

"""
    HMCRandomInputs{T}

Pre-sampled random inputs for one HMC step.

# Fields
- `r`: Initial momentum (D-dimensional)
- `u`: Uniform random for MH accept-reject (scalar in [0,1])
"""
struct HMCRandomInputs{T<:AbstractFloat,V<:AbstractVector{T}}
    r::V      # Initial momentum
    u::T      # Uniform for accept-reject
end

"""
    sample_hmc_inputs(rng, D, T_len; M⁻¹=ones(D))

Sample random inputs for T_len HMC steps.

# Arguments
- `rng`: Random number generator
- `D`: Dimension
- `T_len`: Number of HMC steps
- `M⁻¹`: Inverse mass matrix diagonal (for momentum sampling)
"""
function sample_hmc_inputs(
    rng::AbstractRNG,
    D::Int,
    T_len::Int;
    M⁻¹::AbstractVector=ones(D),
)
    # Momentum is sampled from N(0, M), so r ~ N(0, M) means r = sqrt(M) * z where z ~ N(0, I)
    # For diagonal M, sqrt(M) = 1/sqrt(M⁻¹)
    sqrt_M = 1 ./ sqrt.(M⁻¹)

    return [HMCRandomInputs(
        sqrt_M .* randn(rng, D),  # r ~ N(0, M)
        rand(rng),                 # u ~ Uniform(0, 1)
    ) for _ in 1:T_len]
end

####
#### Leapfrog Integration
####

"""
    leapfrog_step(θ, r, ∇logp, ε; M⁻¹=ones(length(θ)))

Perform one leapfrog step.

# Arguments
- `θ`: Position (D-dimensional)
- `r`: Momentum (D-dimensional)
- `∇logp`: Gradient of log density at θ
- `ε`: Step size
- `M⁻¹`: Inverse mass matrix diagonal

# Returns
- `(θ_new, r_new)`: Updated position and momentum
"""
function leapfrog_step(
    θ::AbstractVector{T},
    r::AbstractVector{T},
    ∇logp::AbstractVector{T},
    ε::T;
    M⁻¹::AbstractVector{T}=ones(T, length(θ)),
) where {T}
    # Half step for momentum
    r_half = r .+ (ε / 2) .* ∇logp

    # Full step for position
    θ_new = θ .+ ε .* (M⁻¹ .* r_half)

    # Note: We don't compute the second half-step here because
    # we need the new gradient at θ_new, which the caller provides

    return θ_new, r_half
end

"""
    leapfrog_full(θ, r, ∇logp_fn, ε, L; M⁻¹=ones(length(θ)))

Perform L leapfrog steps (full integration).

# Arguments
- `θ`: Initial position
- `r`: Initial momentum
- `∇logp_fn`: Function computing gradient of log density
- `ε`: Step size
- `L`: Number of leapfrog steps
- `M⁻¹`: Inverse mass matrix diagonal

# Returns
- `(θ_final, r_final)`: Final position and momentum
"""
function leapfrog_full(
    θ::AbstractVector{T},
    r::AbstractVector{T},
    ∇logp_fn,
    ε::T,
    L::Int;
    M⁻¹::AbstractVector{T}=ones(T, length(θ)),
) where {T}
    θ_curr = copy(θ)
    r_curr = copy(r)

    for l in 1:L
        ∇logp = ∇logp_fn(θ_curr)

        # Half step for momentum
        r_curr = r_curr .+ (ε / 2) .* ∇logp

        # Full step for position
        θ_curr = θ_curr .+ ε .* (M⁻¹ .* r_curr)

        # Half step for momentum (with new gradient)
        ∇logp = ∇logp_fn(θ_curr)
        r_curr = r_curr .+ (ε / 2) .* ∇logp
    end

    return θ_curr, r_curr
end

####
#### HMC Transition (Approach A: Parallelize across HMC steps)
####

"""
    HMCConfig{T,F1,F2}

Configuration for HMC sampling.

# Fields
- `ε`: Step size
- `L`: Number of leapfrog steps
- `logp`: Log density function
- `∇logp`: Gradient of log density function
- `M⁻¹`: Inverse mass matrix diagonal
"""
struct HMCConfig{T<:AbstractFloat,V<:AbstractVector{T},F1,F2}
    ε::T
    L::Int
    logp::F1
    ∇logp::F2
    M⁻¹::V
end

function HMCConfig(ε::T, L::Int, logp, ∇logp) where {T}
    # We'll infer dimension from first call
    return HMCConfig(ε, L, logp, ∇logp, T[])
end

function HMCConfig(ε::T, L::Int, logp, ∇logp, D::Int) where {T}
    return HMCConfig(ε, L, logp, ∇logp, ones(T, D))
end

"""
    hmc_proposal(θ, r, ∇logp_fn, ε, L; M⁻¹)

Compute HMC proposal by running leapfrog integration.

# Returns
- `(θ_prop, r_prop)`: Proposed position and (negated) momentum
"""
function hmc_proposal(
    θ::AbstractVector{T},
    r::AbstractVector{T},
    ∇logp_fn,
    ε::T,
    L::Int;
    M⁻¹::AbstractVector{T}=ones(T, length(θ)),
) where {T}
    θ_prop, r_prop = leapfrog_full(θ, r, ∇logp_fn, ε, L; M⁻¹=M⁻¹)
    # Negate momentum for reversibility
    return θ_prop, -r_prop
end

"""
    kinetic_energy(r, M⁻¹)

Compute kinetic energy K(r) = 0.5 * r' * M⁻¹ * r
"""
function kinetic_energy(r::AbstractVector{T}, M⁻¹::AbstractVector{T}) where {T}
    return T(0.5) * sum(M⁻¹ .* r .^ 2)
end

"""
    hmc_log_accept_ratio(θ, r, θ_prop, r_prop, logp_fn; M⁻¹)

Compute log acceptance ratio for HMC.

log α = log p(θ') - log p(θ) + K(r) - K(r')
"""
function hmc_log_accept_ratio(
    θ::AbstractVector{T},
    r::AbstractVector{T},
    θ_prop::AbstractVector{T},
    r_prop::AbstractVector{T},
    logp_fn;
    M⁻¹::AbstractVector{T}=ones(T, length(θ)),
) where {T}
    logp_curr = logp_fn(θ)
    logp_prop = logp_fn(θ_prop)
    K_curr = kinetic_energy(r, M⁻¹)
    K_prop = kinetic_energy(r_prop, M⁻¹)

    return logp_prop - logp_curr + K_curr - K_prop
end

"""
    hmc_transition(θ, ω, logp, ∇logp, ε, L; M⁻¹)

One HMC transition step with soft MH gating for DEER compatibility.

# Arguments
- `θ`: Current position
- `ω`: HMCRandomInputs (pre-sampled momentum and uniform)
- `logp`: Log density function
- `∇logp`: Gradient function
- `ε`: Step size
- `L`: Number of leapfrog steps
- `M⁻¹`: Inverse mass matrix diagonal

# Returns
- `θ_new`: New position after MH accept/reject
"""
function hmc_transition(
    θ::AbstractVector{T},
    ω::HMCRandomInputs{T},
    logp,
    ∇logp,
    ε::T,
    L::Int;
    M⁻¹::AbstractVector{T}=ones(T, length(θ)),
) where {T}
    r = ω.r
    u = ω.u

    # Compute proposal via leapfrog
    θ_prop, r_prop = hmc_proposal(θ, r, ∇logp, ε, L; M⁻¹=M⁻¹)

    # Compute acceptance probability
    log_α = hmc_log_accept_ratio(θ, r, θ_prop, r_prop, logp; M⁻¹=M⁻¹)

    # Soft gating for DEER (differentiable approximation)
    # Use sigmoid with temperature τ → 0 for hard accept/reject
    # For now, use hard accept/reject (will add soft version for gradient)
    accept = log(u) < log_α

    return accept ? θ_prop : θ
end

"""
    hmc_transition_soft(θ, ω, logp, ∇logp, ε, L; M⁻¹, τ=0.1)

Soft-gated HMC transition for DEER compatibility.

Uses a differentiable soft-max gate instead of hard accept/reject.
"""
function hmc_transition_soft(
    θ::AbstractVector{T},
    ω::HMCRandomInputs{T},
    logp,
    ∇logp,
    ε::T,
    L::Int;
    M⁻¹::AbstractVector{T}=ones(T, length(θ)),
    τ::T=T(0.1),
) where {T}
    r = ω.r
    u = ω.u

    # Compute proposal via leapfrog
    θ_prop, r_prop = hmc_proposal(θ, r, ∇logp, ε, L; M⁻¹=M⁻¹)

    # Compute acceptance probability
    log_α = hmc_log_accept_ratio(θ, r, θ_prop, r_prop, logp; M⁻¹=M⁻¹)

    # Soft gating: interpolate between proposal and current
    # g ∈ [0, 1] where g → 1 means accept, g → 0 means reject
    g = sigmoid((log_α - log(u)) / τ)

    return g .* θ_prop .+ (1 - g) .* θ
end

####
#### Sequential HMC (for comparison)
####

"""
    sequential_hmc(logp, ∇logp, ε, L, s0, T_len; kwargs...)

Run HMC sequentially for T_len steps.

# Returns
- `trajectory`: T_len × D matrix of samples
- `acceptance_rate`: Fraction of accepted proposals
"""
function sequential_hmc(
    logp,
    ∇logp,
    ε::T,
    L::Int,
    s0::AbstractVector{T},
    T_len::Int;
    rng::AbstractRNG=default_rng(),
    M⁻¹::AbstractVector{T}=ones(T, length(s0)),
) where {T}
    D = length(s0)
    trajectory = zeros(T, T_len, D)

    # Sample random inputs
    ω = sample_hmc_inputs(rng, D, T_len; M⁻¹=M⁻¹)

    θ_curr = copy(s0)
    n_accept = 0

    for t in 1:T_len
        θ_new = hmc_transition(θ_curr, ω[t], logp, ∇logp, ε, L; M⁻¹=M⁻¹)

        # Track acceptance
        if θ_new != θ_curr
            n_accept += 1
        end

        trajectory[t, :] = θ_new
        θ_curr = θ_new
    end

    return trajectory, n_accept / T_len
end

"""
    sequential_hmc(config::HMCConfig, s0, T_len, ω)

Run HMC with pre-sampled random inputs.
"""
function sequential_hmc(
    config::HMCConfig{T},
    s0::AbstractVector{T},
    T_len::Int,
    ω::Vector{<:HMCRandomInputs{T}},
) where {T}
    D = length(s0)
    M⁻¹ = isempty(config.M⁻¹) ? ones(T, D) : config.M⁻¹

    trajectory = zeros(T, T_len, D)
    θ_curr = copy(s0)
    n_accept = 0

    for t in 1:T_len
        θ_new = hmc_transition(θ_curr, ω[t], config.logp, config.∇logp, config.ε, config.L; M⁻¹=M⁻¹)

        if !all(θ_new .≈ θ_curr)
            n_accept += 1
        end

        trajectory[t, :] = θ_new
        θ_curr = θ_new
    end

    return trajectory, n_accept / T_len
end

####
#### Parallel HMC (Approach A: Parallelize across HMC steps)
####

"""
    parallel_hmc(config::HMCConfig, s0, T_len, ω; method=QuasiDEER(), kwargs...)

Run HMC in parallel using DEER across T_len steps.

This is Approach A: treat each full HMC step as a transition function
and parallelize across the T samples.

# Arguments
- `config`: HMCConfig with step size, leapfrog steps, log density, etc.
- `s0`: Initial state
- `T_len`: Number of HMC steps
- `ω`: Pre-sampled random inputs

# Returns
- `DEERResult` containing the trajectory and convergence info
"""
function parallel_hmc(
    config::HMCConfig{T},
    s0::AbstractVector{T},
    T_len::Int,
    ω::Vector{<:HMCRandomInputs{T}};
    method::AbstractParallelMethod=QuasiDEER(),
    kwargs...,
) where {T}
    D = length(s0)
    M⁻¹ = isempty(config.M⁻¹) ? ones(T, D) : config.M⁻¹

    # Create transition function for DEER
    # Note: Using soft gating for differentiability
    f(θ, ω_t) = hmc_transition_soft(
        θ, ω_t, config.logp, config.∇logp, config.ε, config.L;
        M⁻¹=M⁻¹, τ=T(0.01)  # Small τ for near-hard gating
    )

    return deer(f, s0, T_len, ω; method=method, kwargs...)
end

"""
    parallel_hmc(logp, ∇logp, ε, L, s0, T_len; kwargs...)

Convenience API for parallel HMC.
"""
function parallel_hmc(
    logp,
    ∇logp,
    ε::T,
    L::Int,
    s0::AbstractVector{T},
    T_len::Int;
    rng::AbstractRNG=default_rng(),
    M⁻¹::AbstractVector{T}=ones(T, length(s0)),
    method::AbstractParallelMethod=QuasiDEER(),
    kwargs...,
) where {T}
    D = length(s0)
    ω = sample_hmc_inputs(rng, D, T_len; M⁻¹=M⁻¹)
    config = HMCConfig(ε, L, logp, ∇logp, M⁻¹)

    return parallel_hmc(config, s0, T_len, ω; method=method, kwargs...)
end

####
#### Approach B: Parallelize Leapfrog Steps (Block Quasi-DEER)
####
#### This approach parallelizes the L leapfrog steps within a single HMC step.
#### The state is s = [θ; r] and the Jacobian has 2×2 block structure.
####

"""
    LeapfrogRandomInputs{T}

Random inputs for leapfrog (actually deterministic - no randomness in leapfrog).
This struct is for API consistency with DEER.
"""
struct LeapfrogRandomInputs{T<:AbstractFloat}
    # Leapfrog is deterministic given initial conditions
    # This struct exists for API consistency
    dummy::T
end

"""
    leapfrog_transition(s, ω, ∇logp, ε, M⁻¹)

One leapfrog step as a transition function for DEER.

State s = [θ; r] is 2D-dimensional.

# Arguments
- `s`: State [θ; r] (2D-dimensional)
- `ω`: Unused (leapfrog is deterministic)
- `∇logp`: Gradient function
- `ε`: Step size
- `M⁻¹`: Inverse mass matrix diagonal

# Returns
- `s_new`: New state [θ'; r']
"""
function leapfrog_transition(
    s::AbstractVector{T},
    ω,  # Unused but needed for DEER interface
    ∇logp,
    ε::T,
    M⁻¹::AbstractVector{T},
) where {T}
    D = length(s) ÷ 2
    θ = s[1:D]
    r = s[(D+1):end]

    # Get gradient at current position
    grad = ∇logp(θ)

    # Half step for momentum
    r = r .+ (ε / 2) .* grad

    # Full step for position
    θ = θ .+ ε .* (M⁻¹ .* r)

    # Half step for momentum (with new gradient)
    grad = ∇logp(θ)
    r = r .+ (ε / 2) .* grad

    return vcat(θ, r)
end

"""
    parallel_leapfrog(θ0, r0, ∇logp, ε, L, M⁻¹; method=QuasiDEER(), kwargs...)

Parallelize L leapfrog steps using DEER.

This is Approach B: parallelize within a single HMC proposal.

# Arguments
- `θ0`: Initial position
- `r0`: Initial momentum
- `∇logp`: Gradient function
- `ε`: Step size
- `L`: Number of leapfrog steps
- `M⁻¹`: Inverse mass matrix diagonal

# Returns
- `DEERResult` containing the trajectory of [θ; r] states
"""
function parallel_leapfrog(
    θ0::AbstractVector{T},
    r0::AbstractVector{T},
    ∇logp,
    ε::T,
    L::Int,
    M⁻¹::AbstractVector{T};
    method::AbstractParallelMethod=QuasiDEER(),
    kwargs...,
) where {T}
    D = length(θ0)

    # Initial state
    s0 = vcat(θ0, r0)

    # Dummy random inputs (leapfrog is deterministic)
    ω = [LeapfrogRandomInputs(zero(T)) for _ in 1:L]

    # Transition function
    f(s, ω_t) = leapfrog_transition(s, ω_t, ∇logp, ε, M⁻¹)

    return deer(f, s0, L, ω; method=method, kwargs...)
end

####
#### Block Quasi-DEER for Leapfrog
####
#### The leapfrog Jacobian has a specific 2×2 block structure per dimension:
####
####   J = [ I_D          ε*M⁻¹        ]
####       [ ε*diag(H)    I_D + ε²*M⁻¹*diag(H) ]
####
#### where H = -∇²logp(θ) is the Hessian.
####

"""
    leapfrog_block_jacobian(θ, ∇logp, hessian_diag_fn, ε, M⁻¹)

Compute the 2×2 block Jacobian of one leapfrog step.

# Returns
- `J11, J12, J21, J22`: D×D diagonal blocks (stored as D-vectors for diagonal case)
"""
function leapfrog_block_jacobian(
    θ::AbstractVector{T},
    ∇logp,
    hessian_diag_fn,  # Function to compute diagonal of Hessian
    ε::T,
    M⁻¹::AbstractVector{T},
) where {T}
    D = length(θ)

    # Get Hessian diagonal at current and final positions
    # For simplicity, we use the Hessian at the initial position
    # (This is an approximation; the exact Jacobian is more complex)
    H_diag = hessian_diag_fn(θ)  # Negative Hessian of log p

    # Block Jacobian structure (for diagonal mass matrix):
    # J = [ I          ε*M⁻¹        ]
    #     [ ε*H_diag   I + ε²*M⁻¹*H_diag ]

    J11 = ones(T, D)           # I_D (diagonal)
    J12 = ε .* M⁻¹             # ε*M⁻¹ (diagonal)
    J21 = ε .* H_diag          # ε*H_diag (diagonal)
    J22 = ones(T, D) .+ ε^2 .* M⁻¹ .* H_diag  # I + ε²*M⁻¹*H_diag

    return J11, J12, J21, J22
end

"""
    hessian_diagonal_fd(∇logp, θ; δ=1e-5)

Estimate Hessian diagonal via finite differences on the gradient.
"""
function hessian_diagonal_fd(
    ∇logp,
    θ::AbstractVector{T};
    δ::T=T(1e-5),
) where {T}
    D = length(θ)
    H_diag = zeros(T, D)
    grad0 = ∇logp(θ)

    for i in 1:D
        θ_plus = copy(θ)
        θ_plus[i] += δ
        grad_plus = ∇logp(θ_plus)
        H_diag[i] = (grad_plus[i] - grad0[i]) / δ
    end

    return H_diag
end
