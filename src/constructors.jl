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

Usage:

```julia
NUTS()            # Use default NUTS configuration.
NUTS(1000, 0.65)  # Use 1000 adaption steps, and target accept ratio 0.65.
```

Arguments:

- `n_adapts::Int` : The number of samples to use with adaptation.
- `δ::Real` : Target acceptance rate for dual averaging.
- `max_depth::Int` : Maximum doubling tree depth.
- `Δ_max::Real` : Maximum divergence during doubling tree.
- `init_ϵ::Real` : Initial step size; 0 means automatically searching using a heuristic procedure.

"""
Base.@kwdef struct NUTS{T<:AbstractFloat} <: AbstractHMCSampler
    n_adapts::Int
    δ::T
    max_depth::Int = 10
    Δ_max::T = T(1000)
    init_ϵ::T = zero(T)
    integrator_method = Leapfrog
    metric_type = DiagEuclideanMetric
end

###########
### HMC ###
###########
"""
    HMC(ϵ::Real, n_leapfrog::Int)

Hamiltonian Monte Carlo sampler with static trajectory.

Arguments:

- `ϵ::Real` : The leapfrog step size to use.
- `n_leapfrog::Int` : The number of leapfrog steps to use.

Usage:

```julia
HMC(0.05, 10)
```

Tips:

- If you are receiving gradient errors when using `HMC`, try reducing the leapfrog step size `ϵ`, e.g.

```julia
# Original step size
sample(gdemo([1.5, 2]), HMC(0.1, 10), 1000)

# Reduced step size
sample(gdemo([1.5, 2]), HMC(0.01, 10), 1000)
```
"""
Base.@kwdef struct HMC{T<:AbstractFloat} <: AbstractHMCSampler
    init_ϵ::T
    n_leapfrog::Int
    integrator_method = Leapfrog
    metric_type = DiagEuclideanMetric
end

#############
### HMCDA ###
#############
"""
    HMCDA(n_adapts::Int, δ::Real, λ::Real; ϵ::Real=0)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(200, 0.65, 0.3)
```

Arguments:

- `n_adapts::Int` : Numbers of samples to use for adaptation.
- `δ::Real` : Target acceptance rate. 65% is often recommended.
- `λ::Real` : Target leapfrog length.
- `ϵ::Real=0` : Initial step size. If 0, then it is automatically determined.

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

- Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
  setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
  Research 15, no. 1 (2014): 1593-1623.
"""
Base.@kwdef struct HMCDA{T<:AbstractFloat} <: AbstractHMCSampler
    n_adapts::Int
    δ::T
    λ::T
    init_ϵ::T = zero(T)
    integrator_method = Leapfrog
    metric_type = DiagEuclideanMetric
end

export HMCSampler, HMC, NUTS, HMCDA
