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
end

function HMCSampler(κ, metric, adaptor)
    T = collect(typeof(metric).parameters)[1]
    return HMCSampler{T}(κ, metric, adaptor)
end

############
### NUTS ###
############
"""
    NUTS(δ::Real; max_depth::Int=10, Δ_max::Real=1000, init_ϵ::Real=0, integrator = :leapfrog, metric = :diagonal)

No-U-Turn Sampler (NUTS) sampler.

# Fields

$(FIELDS)

# Usage:

```julia
NUTS(δ=0.65)  # Use target accept ratio 0.65.
```
"""
struct NUTS{T<:Real} <: AbstractHMCSampler{T}
    "Target acceptance rate for dual averaging."
    δ::T
    "Maximum doubling tree depth."
    max_depth::Int
    "Maximum divergence during doubling tree."
    Δ_max::T
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::Union{Symbol,AbstractIntegrator}
    "Choice of initial metric;  `Symbol` means it is automatically initialised. The metric type will be preserved during automatic initialisation and adaption."
    metric::Union{Symbol,AbstractMetric}
end

function NUTS(δ; max_depth = 10, Δ_max = 1000.0, integrator = :leapfrog, metric = :diagonal)
    T = typeof(δ)
    return NUTS(δ, max_depth, T(Δ_max), integrator, metric)
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
HMC(init_ϵ=0.05, n_leapfrog=10, integrator = :leapfrog, metric = :diagonal)
```
"""
struct HMC{T<:Real} <: AbstractHMCSampler{T}
    "Number of leapfrog steps."
    n_leapfrog::Int
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::Union{Symbol,AbstractIntegrator}
    "Choice of initial metric;  `Symbol` means it is automatically initialised. The metric type will be preserved during automatic initialisation and adaption."
    metric::Union{Symbol,AbstractMetric}
end

function HMC(n_leapfrog; integrator = :leapfrog, metric = :diagonal)
    if integrator isa Symbol
        T = typeof(0.0) # current default float type
    else
        T = integrator_eltype(integrator)
    end
    return HMC{T}(n_leapfrog, integrator, metric)
end

#############
### HMCDA ###
#############
"""
    HMCDA(δ::Real, λ::Real; ϵ::Real=0, integrator = :leapfrog, metric = :diagonal)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

# Fields

$(FIELDS)

# Usage:

```julia
HMCDA(δ=0.65, λ=0.3)
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
    "Choice of integrator, specified either using a `Symbol` or [`AbstractIntegrator`](@ref)"
    integrator::Union{Symbol,AbstractIntegrator}
    "Choice of initial metric;  `Symbol` means it is automatically initialised. The metric type will be preserved during automatic initialisation and adaption."
    metric::Union{Symbol,AbstractMetric}
end

function HMCDA(δ, λ; init_ϵ = 0, integrator = :leapfrog, metric = :diagonal)
    δ, λ = promote(δ, λ)
    T = typeof(δ)
    return HMCDA(δ, T(λ), integrator, metric)
end
