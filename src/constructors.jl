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
struct HMCSampler{
    I<:AbstractIntegrator,
    K<:AbstractMCMCKernel,
    M<:AbstractMetric,
    A<:Adaptation.AbstractAdaptor,
} <: AbstractHMCSampler
    "[`integrator`](@ref)."
    integrator::I
    "[`AbstractMCMCKernel`](@ref)."
    kernel::K
    "[`AbstractMetric`](@ref)."
    metric::M
    "[`AbstractAdaptor`](@ref)."
    adaptor::A
    "Adaptation steps if any"
    n_adapts::Int
end

HMCSampler(kernel, metric, adaptor; n_adapts = 0) =
    HMCSampler(Leapfrog, kernel, metric, adaptor, n_adapts)

############
### NUTS ###
############
"""
    NUTS(n_adapts::Int, δ::Float64; max_depth::Int=10, Δ_max::Float64=1000.0, init_ϵ::Float64=0.0)

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS()            # Use default NUTS configuration.
NUTS(1000, 0.65)  # Use 1000 adaption steps, and target accept ratio 0.65.
```

Arguments:

- `n_adapts::Int` : The number of samples to use with adaptation.
- `δ::Float64` : Target acceptance rate for dual averaging.
- `max_depth::Int` : Maximum doubling tree depth.
- `Δ_max::Float64` : Maximum divergence during doubling tree.
- `init_ϵ::Float64` : Initial step size; 0 means automatically searching using a heuristic procedure.

"""
Base.@kwdef struct NUTS <: AbstractHMCSampler
    n_adapts::Int                      # number of samples with adaption for ϵ
    δ::Float64                         # target accept rate
    max_depth::Int = 10                # maximum tree depth
    Δ_max::Float64 = 1000.0            # maximum error
    init_ϵ::Float64 = 0.0              # (initial) step size
    integrator_method = Leapfrog       # integrator method
    metric_type = DiagEuclideanMetric  # metric type
end

###########
### HMC ###
###########
"""
    HMC(ϵ::Float64, n_leapfrog::Int)

Hamiltonian Monte Carlo sampler with static trajectory.

Arguments:

- `ϵ::Float64` : The leapfrog step size to use.
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
Base.@kwdef struct HMC <: AbstractHMCSampler
    init_ϵ::Float64                    # leapfrog step size
    n_leapfrog::Int                    # leapfrog step number
    integrator_method = Leapfrog       # integrator method
    metric_type = DiagEuclideanMetric  # metric type
end

#############
### HMCDA ###
#############
"""
    HMCDA(n_adapts::Int, δ::Float64, λ::Float64; ϵ::Float64=0.0)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(200, 0.65, 0.3)
```

Arguments:

- `n_adapts::Int` : Numbers of samples to use for adaptation.
- `δ::Float64` : Target acceptance rate. 65% is often recommended.
- `λ::Float64` : Target leapfrog length.
- `ϵ::Float64=0.0` : Initial step size; 0 means automatically search by Turing.

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

- Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
  setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
  Research 15, no. 1 (2014): 1593-1623.
"""
Base.@kwdef struct HMCDA <: AbstractHMCSampler
    n_adapts::Int                      # number of samples with adaption for ϵ
    δ::Float64                         # target accept rate
    λ::Float64                         # target leapfrog length
    init_ϵ::Float64 = 0.0              # (initial) step size
    integrator_method = Leapfrog       # integrator method
    metric_type = DiagEuclideanMetric  # metric type
end

export HMCSampler, HMC, NUTS, HMCDA
