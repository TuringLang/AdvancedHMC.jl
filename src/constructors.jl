abstract type AbstractHMCSampler{T<:Real} <: AbstractMCMC.AbstractSampler end

##############
### Custom ###
##############

"""
    HMCSampler

An `AbstractMCMC.AbstractSampler` for kernels in AdvancedHMC.jl.

# Fields

$(FIELDS)

# Notes

Note that all the fields have the prefix `initial_` to indicate
that these will not necessarily correspond to the `kernel`, `metric`,
and `adaptor` after sampling.

To access the updated fields use the resulting [`HMCState`](@ref).
"""
struct HMCSampler{T<:Real} <: AbstractHMCSampler{T}
    "[`AbstractMCMCKernel`](@ref)."
    κ::AbstractMCMCKernel
    "Choice of initial metric [`AbstractMetric`](@ref). The metric type will be preserved during adaption."
    metric::AbstractMetric
    "[`AbstractAdaptor`](@ref)."
    adaptor::AbstractAdaptor
    "Adaptation steps if any"
    n_adapts::Int
end

function HMCSampler(κ, metric, adaptor; n_adapts = 0)
    T = collect(typeof(metric).parameters)[1]
    return HMCSampler{T}(κ, metric, adaptor, n_adapts)
end

############
### NUTS ###
############
"""
    NUTS(n_adapts::Int, δ::Real; max_depth::Int=10, Δ_max::Real=1000, init_ϵ::Real=0)

No-U-Turn Sampler (NUTS) sampler.

# Fields

$(FIELDS)

# Usage:

```julia
NUTS(n_adapts=1000, δ=0.65)  # Use 1000 adaption steps, and target accept ratio 0.65.
```
"""
struct NUTS{T<:Real} <: AbstractHMCSampler{T}
    "Target acceptance rate for dual averaging."
    δ::T
    "Maximum doubling tree depth."
    max_depth::Int
    "Maximum divergence during doubling tree."
    Δ_max::T
    "Initial step size; 0 means it is automatically chosen."
    init_ϵ::T
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::Union{Symbol,AbstractIntegrator}
    "Choice of initial metric, specified using a `Symbol` or `AbstractMetric`. The metric type will be preserved during adaption."
    metric::Union{Symbol,AbstractMetric}
end

function NUTS(
    δ;
    max_depth = 10,
    Δ_max = 1000.0,
    init_ϵ = 0.0,
    integrator = :leapfrog,
    metric = :diagonal,
)
    T = typeof(δ)
    return NUTS(δ, max_depth, T(Δ_max), T(init_ϵ), integrator, metric)
end

###########
### HMC ###
###########
"""
    HMC(ϵ::Real, n_leapfrog::Int)

Hamiltonian Monte Carlo sampler with static trajectory.

# Fields

$(FIELDS)

# Usage:

```julia
HMC(init_ϵ=0.05, n_leapfrog=10)
```
"""
struct HMC{T<:Real} <: AbstractHMCSampler{T}
    "Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T
    "Number of leapfrog steps."
    n_leapfrog::Int
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::Union{Symbol,AbstractIntegrator}
    "Choice of initial metric, specified using a `Symbol` or `AbstractMetric`. The metric type will be preserved during adaption."
    metric::Union{Symbol,AbstractMetric}
end

function HMC(init_ϵ, n_leapfrog; integrator = :leapfrog, metric = :diagonal)
    return HMC(init_ϵ, n_leapfrog, integrator, metric)
end

#############
### HMCDA ###
#############
"""
    HMCDA(n_adapts::Int, δ::Real, λ::Real; ϵ::Real=0)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

# Fields

$(FIELDS)

# Usage:

```julia
HMCDA(n_adapts=200, δ=0.65, λ=0.3)
```

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

- Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
  setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
  Research 15, no. 1 (2014): 1593-1623.
"""
struct HMCDA{T<:Real} <: AbstractHMCSampler{T}
    "Target acceptance rate for dual averaging."
    δ::T
    "Target leapfrog length."
    λ::T
    "Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::Union{Symbol,AbstractIntegrator}
    "Choice of initial metric, specified using a `Symbol` or `AbstractMetric`. The metric type will be preserved during adaption."
    metric::Union{Symbol,AbstractMetric}
end

function HMCDA(δ, λ; init_ϵ = 0.0, integrator = :leapfrog, metric = :diagonal)
    if typeof(δ) != typeof(λ)
        @warn "typeof(δ) != typeof(λ) --> using typeof(δ)"
    end
    T = typeof(δ)
    return HMCDA(δ, T(λ), T(init_ϵ), integrator, metric)
end
