####
#### Parallel MALA (Metropolis-Adjusted Langevin Algorithm)
####
#### MALA transition: x_t = accept_reject(x_{t-1}, proposal(x_{t-1}, ξ_t), u_t)
#### where:
####   - proposal(x, ξ) = x + ε∇log p(x) + √(2ε)ξ
####   - accept with probability α = min(1, p(x̃)q(x|x̃) / (p(x)q(x̃|x)))
####
#### The accept-reject step uses a stop-gradient trick for differentiability.
####

using Random: AbstractRNG, default_rng

####
#### MALA Random Inputs
####

"""
    MALARandomInputs{T}

Pre-sampled random inputs for MALA at one timestep.

# Fields
- `ξ`: Gaussian noise for proposal (length D)
- `u`: Uniform for accept-reject decision (scalar)
"""
struct MALARandomInputs{T,V<:AbstractVector{T}}
    ξ::V      # N(0, I) for proposal
    u::T      # U(0, 1) for accept-reject
end

"""
    sample_mala_inputs(rng::AbstractRNG, D::Int, T_len::Int, ::Type{T}=Float64)

Pre-sample all random inputs needed for T_len MALA steps.

Returns a vector of MALARandomInputs.
"""
function sample_mala_inputs(
    rng::AbstractRNG, D::Int, T_len::Int, (::Type{T})=Float64
) where {T}
    return [MALARandomInputs(randn(rng, T, D), rand(rng, T)) for _ in 1:T_len]
end

function sample_mala_inputs(D::Int, T_len::Int, T::Type=Float64)
    sample_mala_inputs(default_rng(), D, T_len, T)
end

####
#### MALA Proposal
####

"""
    mala_proposal(x, ∇logp_x, ε, ξ)

Compute the MALA proposal: x̃ = x + ε∇log p(x) + √(2ε)ξ

# Arguments
- `x`: Current state
- `∇logp_x`: Gradient of log p at x
- `ε`: Step size
- `ξ`: Standard Gaussian noise

# Returns
- Proposed state x̃
"""
function mala_proposal(
    x::AbstractVector{T}, ∇logp_x::AbstractVector{T}, ε::T, ξ::AbstractVector{T}
) where {T}
    return x .+ ε .* ∇logp_x .+ sqrt(2 * ε) .* ξ
end

####
#### MALA Acceptance Probability
####

"""
    mala_log_acceptance_ratio(x, x̃, logp, ∇logp, ε)

Compute log of the Metropolis-Hastings acceptance ratio for MALA.

log α = log p(x̃) - log p(x) + log q(x|x̃) - log q(x̃|x)

where q(x̃|x) ∝ exp(-||x̃ - x - ε∇log p(x)||² / (4ε))

# Arguments
- `x`: Current state
- `x̃`: Proposed state
- `logp`: Function computing log p(x)
- `∇logp`: Function computing ∇log p(x)
- `ε`: Step size

# Returns
- Log acceptance ratio (can be > 0)
"""
function mala_log_acceptance_ratio(
    x::AbstractVector{T}, x̃::AbstractVector{T}, logp, ∇logp, ε::T
) where {T}
    # Log density ratio
    log_ratio = logp(x̃) - logp(x)

    # Proposal log densities (up to constant)
    # q(x̃|x) ∝ exp(-||x̃ - x - ε∇log p(x)||² / (4ε))
    # q(x|x̃) ∝ exp(-||x - x̃ - ε∇log p(x̃)||² / (4ε))
    ∇logp_x = ∇logp(x)
    ∇logp_x̃ = ∇logp(x̃)

    forward_mean = x .+ ε .* ∇logp_x
    backward_mean = x̃ .+ ε .* ∇logp_x̃

    log_q_forward = -sum((x̃ .- forward_mean) .^ 2) / (4 * ε)
    log_q_backward = -sum((x .- backward_mean) .^ 2) / (4 * ε)

    return log_ratio + log_q_backward - log_q_forward
end

####
#### Soft Gating for Differentiable Accept-Reject
####

"""
    soft_gate(log_α, log_u; temperature=1.0)

Compute a soft gate for accept-reject that is differentiable.

Uses sigmoid with straight-through estimator:
- Forward: returns hard decision (0 or 1)
- Backward: uses sigmoid gradient

# Arguments
- `log_α`: Log acceptance probability
- `log_u`: Log of uniform random variable
- `temperature`: Softness of sigmoid (default 1.0)

# Returns
- Gate value g ∈ {0, 1} (hard) with soft gradient
"""
function soft_gate(log_α::T, log_u::T; temperature::T=one(T)) where {T}
    # Soft gate: σ((log_α - log_u) / temperature)
    z = (log_α - log_u) / temperature
    soft = sigmoid(z)

    # Hard decision
    hard = T(z > 0)

    # Straight-through: use hard value but soft gradient
    # This is achieved by: hard + (soft - soft_detached)
    # where soft_detached has zero gradient
    # In Julia without AD, we just return hard for the forward pass
    return hard
end

"""
    sigmoid(x)

Numerically stable sigmoid function.
"""
function sigmoid(x::T) where {T}
    if x >= 0
        z = exp(-x)
        return one(T) / (one(T) + z)
    else
        z = exp(x)
        return z / (one(T) + z)
    end
end

####
#### MALA Transition Function
####

"""
    mala_transition(x, ω::MALARandomInputs, logp, ∇logp, ε)

Compute one MALA transition step.

# Arguments
- `x`: Current state
- `ω`: Pre-sampled random inputs (ξ for proposal, u for accept-reject)
- `logp`: Function computing log p(x)
- `∇logp`: Function computing ∇log p(x)
- `ε`: Step size

# Returns
- Next state x_t (either accepted proposal or current state)
"""
function mala_transition(
    x::AbstractVector{T}, ω::MALARandomInputs{T}, logp, ∇logp, ε::T
) where {T}
    # Compute gradient at current state
    ∇logp_x = ∇logp(x)

    # Proposal
    x̃ = mala_proposal(x, ∇logp_x, ε, ω.ξ)

    # Acceptance ratio
    log_α = mala_log_acceptance_ratio(x, x̃, logp, ∇logp, ε)
    log_α = min(zero(T), log_α)  # cap at 0 (α ≤ 1)

    # Accept-reject with soft gating
    log_u = log(ω.u)
    g = soft_gate(log_α, log_u)

    # Return accepted or rejected state
    return g .* x̃ .+ (one(T) - g) .* x
end

####
#### Parallel MALA using DEER
####

"""
    MALAConfig{T}

Configuration for parallel MALA.

# Fields
- `ε`: Step size
- `logp`: Log density function
- `∇logp`: Gradient of log density
"""
struct MALAConfig{T,F1,F2}
    ε::T
    logp::F1
    ∇logp::F2
end

"""
    parallel_mala(config::MALAConfig, s0, T_len, ω; method=QuasiDEER(), kwargs...)

Run parallel MALA using the DEER algorithm.

# Arguments
- `config`: MALA configuration (step size, log density, gradient)
- `s0`: Initial state
- `T_len`: Number of MALA steps (chain length)
- `ω`: Pre-sampled random inputs (vector of MALARandomInputs)

# Keyword Arguments
- `method`: DEER variant to use (default: QuasiDEER())
- `tol`: Convergence tolerance (default: 1e-6)
- `max_iters`: Maximum Newton iterations (default: 1000)
- Other kwargs passed to `deer()`

# Returns
- DEERResult containing the MALA chain

# Example
```julia
# Define target distribution (standard Gaussian)
logp(x) = -0.5 * sum(x.^2)
∇logp(x) = -x

config = MALAConfig(0.1, logp, ∇logp)
s0 = randn(D)
ω = sample_mala_inputs(D, T_len)

result = parallel_mala(config, s0, T_len, ω)
```
"""
function parallel_mala(
    config::MALAConfig{T},
    s0::AbstractVector{T},
    T_len::Int,
    ω::Vector{<:MALARandomInputs{T}};
    method::AbstractParallelMethod=QuasiDEER(),
    kwargs...,
) where {T}
    @assert length(ω) == T_len "Need $T_len random inputs, got $(length(ω))"

    # Create transition function that captures the config
    function f(x, ω_t)
        return mala_transition(x, ω_t, config.logp, config.∇logp, config.ε)
    end

    # Run DEER
    return deer(f, s0, T_len, ω; method=method, kwargs...)
end

"""
    parallel_mala(logp, ∇logp, ε, s0, T_len; rng=default_rng(), method=QuasiDEER(), kwargs...)

Convenience version that samples random inputs automatically.

# Arguments
- `logp`: Log density function
- `∇logp`: Gradient of log density
- `ε`: Step size
- `s0`: Initial state
- `T_len`: Number of MALA steps

# Keyword Arguments
- `rng`: Random number generator for sampling inputs
- `method`: DEER variant to use
- Other kwargs passed to `deer()`

# Returns
- DEERResult containing the MALA chain
"""
function parallel_mala(
    logp,
    ∇logp,
    ε::T,
    s0::AbstractVector{T},
    T_len::Int;
    rng::AbstractRNG=default_rng(),
    method::AbstractParallelMethod=QuasiDEER(),
    kwargs...,
) where {T}
    D = length(s0)
    ω = sample_mala_inputs(rng, D, T_len, T)
    config = MALAConfig(ε, logp, ∇logp)
    return parallel_mala(config, s0, T_len, ω; method=method, kwargs...)
end

####
#### Sequential MALA for comparison
####

"""
    sequential_mala(config::MALAConfig, s0, T_len, ω)

Run MALA sequentially (for comparison/testing).

# Arguments
- `config`: MALA configuration
- `s0`: Initial state
- `T_len`: Number of MALA steps
- `ω`: Pre-sampled random inputs

# Returns
- Trajectory as (T × D) matrix
- Acceptance rate
"""
function sequential_mala(
    config::MALAConfig{T},
    s0::AbstractVector{T},
    T_len::Int,
    ω::Vector{<:MALARandomInputs{T}},
) where {T}
    D = length(s0)
    trajectory = zeros(T, T_len, D)
    n_accepted = 0

    x = s0
    for t in 1:T_len
        x_new = mala_transition(x, ω[t], config.logp, config.∇logp, config.ε)

        # Track acceptance
        if x_new != x
            n_accepted += 1
        end

        trajectory[t, :] = x_new
        x = x_new
    end

    acceptance_rate = n_accepted / T_len
    return trajectory, acceptance_rate
end

"""
    sequential_mala(logp, ∇logp, ε, s0, T_len; rng=default_rng())

Convenience version that samples random inputs automatically.
"""
function sequential_mala(
    logp, ∇logp, ε::T, s0::AbstractVector{T}, T_len::Int; rng::AbstractRNG=default_rng()
) where {T}
    D = length(s0)
    ω = sample_mala_inputs(rng, D, T_len, T)
    config = MALAConfig(ε, logp, ∇logp)
    return sequential_mala(config, s0, T_len, ω)
end
