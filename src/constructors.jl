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
Base.@kwdef struct NUTS{T<:AbstractFloat} <: AbstractHMCSampler
    "`n_adapts::Int` : Number of adaptation steps."
    n_adapts::Int
    "`δ::Real` : Target acceptance rate for dual averaging."
    δ::T
    "`max_depth::Int` : Maximum doubling tree depth."
    max_depth::Int = 10
    "`Δ_max::Real` : Maximum divergence during doubling tree."
    Δ_max::T = T(1000)
    "`init_ϵ::Real` : Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T = zero(T)
    "[`AbstractIntegrator`](@ref)."
    integrator_method::AbstractIntegrator = Leapfrog
    "[`AbstractMetric`](@ref)."
    metric_type::AbstractMetric = DiagEuclideanMetric
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
Base.@kwdef struct HMC{T<:AbstractFloat} <: AbstractHMCSampler
    "`init_ϵ::Real` : Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T
    "`n_leapfrog::Int` : Number of leapfrog steps."
    n_leapfrog::Int
    "[`AbstractIntegrator`](@ref)."
    integrator_method::AbstractIntegrator = Leapfrog
    "[`AbstractMetric`](@ref)."
    metric_type::AbstractMetric = DiagEuclideanMetric
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
Base.@kwdef struct HMCDA{T<:AbstractFloat} <: AbstractHMCSampler
    "`n_adapts::Int` : Number of adaptation steps."
    n_adapts::Int
    "`δ::Real` : Target acceptance rate for dual averaging."
    δ::T
    "`λ::Real` : Target leapfrog length."
    λ::T
    "`init_ϵ::Real` : Initial step size; 0 means automatically searching using a heuristic procedure."
    init_ϵ::T = zero(T)
    "[`AbstractIntegrator`](@ref)."
    integrator_method::AbstractIntegrator = Leapfrog
    "[`AbstractMetric`](@ref)."
    metric_type::AbstractMetric = DiagEuclideanMetric
end
