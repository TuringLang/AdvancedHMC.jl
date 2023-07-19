abstract type AbstractHMCSampler <: AbstractMCMC.AbstractSampler end

##############
### Custom ###
##############

"""
    HMCSampler

A `AbstractMCMC.AbstractSampler` for kernels in AdvancedHMC.jl.

# Fields

$(FIELDS)

# Notes

Note that all the fields have the prefix `initial_` to indicate
that these will not necessarily correspond to the `kernel`, `metric`,
and `adaptor` after sampling.

To access the updated fields use the resulting [`HMCState`](@ref).
"""
Base.@kwdef struct HMCSampler{
    K<:AbstractMCMCKernel,
    M<:AbstractMetric,
    A<:Adaptation.AbstractAdaptor,
} <: AbstractHMCSampler
    "[`AbstractMCMCKernel`](@ref)."
    kernel::K
    "[`AbstractMetric`](@ref)."
    metric::M
    "[`AbstractAdaptor`](@ref)."
    adaptor::A
    "Adaptation steps if any"
    n_adapts::Int = 0
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
struct NUTS{T<:Real,I,D} <: AbstractHMCSampler
    "Number of adaptation steps."
    n_adapts::Int
    "Target acceptance rate for dual averaging."
    δ::T
    "Maximum doubling tree depth."
    max_depth::Int
    "Maximum divergence during doubling tree."
    Δ_max::T
    "Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T
    "Choice of integrator method given as a symbol"
    integrator_method::I
    "Choice of metric type as given a symbol"
    metric_type::D
end

function NUTS(
    n_adapts,
    δ;
    max_depth = 10,
    Δ_max = 1000.0,
    init_ϵ = 0.0,
    integrator_method = :Leapfrog,
    metric_type = :DiagEuclideanMetric,
)   
    T = typeof(δ)
    return NUTS(n_adapts, δ, max_depth, T(Δ_max), T(init_ϵ), integrator_method, metric_type)
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
struct HMC{T<:Real,I,D} <: AbstractHMCSampler
    "Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T
    "Number of leapfrog steps."
    n_leapfrog::Int
    "Choice of integrator method given as a symbol"
    integrator_method::I
    "Choice of metric type as given a symbol"
    metric_type::D
end

function HMC(
    init_ϵ,
    n_leapfrog;
    integrator_method = :Leapfrog,
    metric_type = :DiagEuclideanMetric,
)
    return HMC(init_ϵ, n_leapfrog, integrator_method, metric_type)
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
struct HMCDA{T<:Real,I,D} <: AbstractHMCSampler
    "`Number of adaptation steps."
    n_adapts::Int
    "Target acceptance rate for dual averaging."
    δ::T
    "Target leapfrog length."
    λ::T
    "Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T
    "Choice of integrator method given as a symbol"
    integrator_method::I
    "Choice of metric type as given a symbol"
    metric_type::D
end

function HMCDA(
    n_adapts,
    δ,
    λ;
    init_ϵ = 0.0,
    integrator_method = :Leapfrog,
    metric_type = :DiagEuclideanMetric,
)   
    if typeof(δ) != typeof(λ)
        @warn "typeof(δ) != typeof(λ) --> using typeof(δ)"
    end
    T = typeof(δ)
    return HMCDA(n_adapts, δ, T(λ), T(init_ϵ), integrator_method, metric_type)
end
