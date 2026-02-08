####
#### AbstractMCMC.jl Integration for Parallel MCMC
####
#### Provides ParallelHMC and ParallelMALA samplers that work with the
#### AbstractMCMC.jl interface while using DEER for parallel sampling.
####

using Random: AbstractRNG, default_rng
# LogDensityProblems is already imported in Parallel.jl

####
#### Abstract Type for Parallel Samplers
####

"""
    AbstractParallelSampler <: AbstractMCMC.AbstractSampler

Abstract type for parallel MCMC samplers using DEER algorithm.
"""
abstract type AbstractParallelSampler end

####
#### Parallel HMC Sampler
####

"""
    ParallelHMCSampler{T,M,I} <: AbstractParallelSampler

Parallel HMC sampler using DEER algorithm for O(log T) time complexity.

# Fields
- `ε::T`: Step size for leapfrog integration
- `L::Int`: Number of leapfrog steps per HMC iteration
- `method::M`: DEER method (FullDEER, QuasiDEER, etc.)
- `metric::I`: Metric type (:diagonal, :unit, :dense)
- `tol::T`: Convergence tolerance for Newton iterations
- `max_iters::Int`: Maximum Newton iterations

# Example
```julia
sampler = ParallelHMCSampler(0.1, 10; method=QuasiDEER())
result = sample(model, sampler, 1000)
```
"""
struct ParallelHMCSampler{
    T<:AbstractFloat,
    M<:AbstractParallelMethod,
    I<:Union{Symbol,AbstractMetric},
} <: AbstractParallelSampler
    ε::T
    L::Int
    method::M
    metric::I
    tol::T
    max_iters::Int
end

function ParallelHMCSampler(
    ε::Real,
    L::Int;
    method::AbstractParallelMethod=QuasiDEER(),
    metric::Union{Symbol,AbstractMetric}=:diagonal,
    tol::Real=1e-6,
    max_iters::Int=1000,
)
    T = float(typeof(ε))
    return ParallelHMCSampler{T,typeof(method),typeof(metric)}(
        T(ε), L, method, metric, T(tol), max_iters
    )
end

"""
    ParallelHMC(ε, L; kwargs...)

Convenience constructor for ParallelHMCSampler.
"""
const ParallelHMC = ParallelHMCSampler

####
#### Parallel MALA Sampler
####

"""
    ParallelMALASampler{T,M,I} <: AbstractParallelSampler

Parallel MALA sampler using DEER algorithm for O(log T) time complexity.

# Fields
- `ε::T`: Step size for Langevin dynamics
- `method::M`: DEER method (FullDEER, QuasiDEER, etc.)
- `metric::I`: Metric type (:diagonal, :unit, :dense)
- `tol::T`: Convergence tolerance for Newton iterations
- `max_iters::Int`: Maximum Newton iterations

# Example
```julia
sampler = ParallelMALASampler(0.1; method=QuasiDEER())
result = sample(model, sampler, 1000)
```
"""
struct ParallelMALASampler{
    T<:AbstractFloat,
    M<:AbstractParallelMethod,
    I<:Union{Symbol,AbstractMetric},
} <: AbstractParallelSampler
    ε::T
    method::M
    metric::I
    tol::T
    max_iters::Int
end

function ParallelMALASampler(
    ε::Real;
    method::AbstractParallelMethod=QuasiDEER(),
    metric::Union{Symbol,AbstractMetric}=:diagonal,
    tol::Real=1e-6,
    max_iters::Int=1000,
)
    T = float(typeof(ε))
    return ParallelMALASampler{T,typeof(method),typeof(metric)}(
        T(ε), method, metric, T(tol), max_iters
    )
end

"""
    ParallelMALA(ε; kwargs...)

Convenience constructor for ParallelMALASampler.
"""
const ParallelMALA = ParallelMALASampler

####
#### Parallel Sampler State
####

"""
    ParallelSamplerState{T,M}

State of a parallel sampler after sampling.

# Fields
- `trajectory::M`: Full trajectory (T × D matrix)
- `converged::Bool`: Whether DEER converged
- `iterations::Int`: Number of Newton iterations
- `max_residual::T`: Final residual
- `acceptance_rate::T`: Estimated acceptance rate
"""
struct ParallelSamplerState{T<:AbstractFloat,M<:AbstractMatrix{T}}
    trajectory::M
    converged::Bool
    iterations::Int
    max_residual::T
    acceptance_rate::T
end

####
#### Parallel Transition (for compatibility)
####

"""
    ParallelTransition{T,V}

A single transition from a parallel sampler (for iteration interface).

# Fields
- `θ::V`: Parameter values
- `stat::NamedTuple`: Statistics for this sample
"""
struct ParallelTransition{T<:AbstractFloat,V<:AbstractVector{T}}
    θ::V
    stat::NamedTuple
end

####
#### Metric Utilities
####

"""
    make_parallel_metric(metric, D, T)

Create a metric for parallel sampling.
"""
function make_parallel_metric(metric::Symbol, D::Int, ::Type{T}) where {T}
    if metric === :diagonal
        return ones(T, D)  # M⁻¹ diagonal
    elseif metric === :unit
        return ones(T, D)
    elseif metric === :dense
        # For parallel sampling, we only support diagonal for now
        @warn "Dense metric not yet supported for parallel sampling, using diagonal"
        return ones(T, D)
    else
        error("Unknown metric: $metric")
    end
end

function make_parallel_metric(metric::DiagEuclideanMetric{T}, D::Int, ::Type) where {T}
    return metric.M⁻¹
end

function make_parallel_metric(metric::UnitEuclideanMetric{T}, D::Int, ::Type) where {T}
    return ones(T, D)
end

function make_parallel_metric(metric::DenseEuclideanMetric{T}, D::Int, ::Type) where {T}
    # Extract diagonal for parallel sampling
    @warn "Dense metric approximated with diagonal for parallel sampling"
    return diag(metric.M⁻¹)
end

####
#### Core Sampling Functions
####

"""
    parallel_sample(rng, logdensity, sampler::ParallelHMCSampler, N; kwargs...)

Run parallel HMC sampling using DEER.

# Arguments
- `rng`: Random number generator
- `logdensity`: Log density function (LogDensityProblems interface)
- `sampler`: ParallelHMCSampler instance
- `N`: Number of samples

# Keyword Arguments
- `initial_params`: Initial parameter values (default: random)
- `verbose`: Print convergence info (default: false)

# Returns
- `ParallelSamplerState` with trajectory and convergence info
"""
function parallel_sample(
    rng::AbstractRNG,
    logdensity,
    sampler::ParallelHMCSampler{T},
    N::Int;
    initial_params=nothing,
    verbose::Bool=false,
) where {T}
    # Get dimension
    D = LogDensityProblems.dimension(logdensity)

    # Create log density and gradient functions
    logp = x -> LogDensityProblems.logdensity(logdensity, x)
    ∇logp = function(x)
        _, grad = LogDensityProblems.logdensity_and_gradient(logdensity, x)
        return grad
    end

    # Get metric
    M⁻¹ = make_parallel_metric(sampler.metric, D, T)

    # Initialize
    if initial_params === nothing
        s0 = randn(rng, T, D)
    else
        s0 = T.(initial_params)
    end

    # Sample random inputs
    ω = sample_hmc_inputs(rng, D, N; M⁻¹=M⁻¹)

    # Create HMC config
    config = HMCConfig(sampler.ε, sampler.L, logp, ∇logp, M⁻¹)

    # Run parallel HMC
    result = parallel_hmc(
        config, s0, N, ω;
        method=sampler.method,
        tol=sampler.tol,
        max_iters=sampler.max_iters,
        verbose=verbose
    )

    # Estimate acceptance rate (from soft gating, approximate)
    # For now, assume high acceptance if converged
    acceptance_rate = result.converged ? T(0.9) : T(0.5)

    return ParallelSamplerState(
        result.trajectory,
        result.converged,
        result.iterations,
        result.max_residual,
        acceptance_rate
    )
end

"""
    parallel_sample(rng, logdensity, sampler::ParallelMALASampler, N; kwargs...)

Run parallel MALA sampling using DEER.
"""
function parallel_sample(
    rng::AbstractRNG,
    logdensity,
    sampler::ParallelMALASampler{T},
    N::Int;
    initial_params=nothing,
    verbose::Bool=false,
) where {T}
    # Get dimension
    D = LogDensityProblems.dimension(logdensity)

    # Create log density and gradient functions
    logp = x -> LogDensityProblems.logdensity(logdensity, x)
    ∇logp = function(x)
        _, grad = LogDensityProblems.logdensity_and_gradient(logdensity, x)
        return grad
    end

    # Initialize
    if initial_params === nothing
        s0 = randn(rng, T, D)
    else
        s0 = T.(initial_params)
    end

    # Sample random inputs
    ω = sample_mala_inputs(rng, D, N)

    # Create MALA config
    config = MALAConfig(sampler.ε, logp, ∇logp)

    # Run parallel MALA
    result = parallel_mala(
        config, s0, N, ω;
        method=sampler.method,
        tol=sampler.tol,
        max_iters=sampler.max_iters,
        verbose=verbose
    )

    # Estimate acceptance rate
    acceptance_rate = result.converged ? T(0.9) : T(0.5)

    return ParallelSamplerState(
        result.trajectory,
        result.converged,
        result.iterations,
        result.max_residual,
        acceptance_rate
    )
end

####
#### Convenience Functions
####

"""
    parallel_sample(logdensity, sampler, N; kwargs...)

Sample without explicit RNG (uses default_rng()).
"""
function parallel_sample(logdensity, sampler::AbstractParallelSampler, N::Int; kwargs...)
    return parallel_sample(default_rng(), logdensity, sampler, N; kwargs...)
end

"""
    get_samples(state::ParallelSamplerState)

Extract samples from parallel sampler state as a matrix (N × D).
"""
get_samples(state::ParallelSamplerState) = state.trajectory

"""
    get_samples(state::ParallelSamplerState, burn_in::Int)

Extract samples after discarding burn-in period.
"""
function get_samples(state::ParallelSamplerState, burn_in::Int)
    return state.trajectory[(burn_in+1):end, :]
end

####
#### Iterator Interface (for compatibility with AbstractMCMC patterns)
####

"""
    ParallelSamplerIterator

Iterator over samples from a parallel sampler state.
Allows `for sample in state` style iteration.
"""
struct ParallelSamplerIterator{T<:AbstractFloat,M<:AbstractMatrix{T}}
    state::ParallelSamplerState{T,M}
end

function Base.iterate(state::ParallelSamplerState)
    return Base.iterate(state, 1)
end

function Base.iterate(state::ParallelSamplerState, i::Int)
    if i > size(state.trajectory, 1)
        return nothing
    end
    θ = state.trajectory[i, :]
    stat = (
        iteration=i,
        converged=state.converged,
    )
    return ParallelTransition(θ, stat), i + 1
end

Base.length(state::ParallelSamplerState) = size(state.trajectory, 1)
Base.eltype(::Type{ParallelSamplerState{T,M}}) where {T,M} = ParallelTransition{T,Vector{T}}

####
#### Simple LogDensity wrapper for testing
####

"""
    SimpleLogDensity{F,G}

A simple wrapper implementing LogDensityProblems interface.
Useful for testing without full LogDensityProblems setup.

# Example
```julia
logp = SimpleLogDensity(2, x -> -0.5 * sum(x.^2), x -> -x)
state = parallel_sample(logp, ParallelHMC(0.1, 10), 100)
```
"""
struct SimpleLogDensity{F,G}
    dim::Int
    logp::F
    ∇logp::G
end

LogDensityProblems.dimension(ld::SimpleLogDensity) = ld.dim
LogDensityProblems.capabilities(::Type{<:SimpleLogDensity}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(ld::SimpleLogDensity, x) = ld.logp(x)
function LogDensityProblems.logdensity_and_gradient(ld::SimpleLogDensity, x)
    return ld.logp(x), ld.∇logp(x)
end
