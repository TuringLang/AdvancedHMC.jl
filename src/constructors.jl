abstract type StaticHamiltonian <: AbstractMCMC.AbstractSampler end
abstract type AdaptiveHamiltonian <: AbstractMCMC.AbstractSampler end

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
struct HMCSampler{K,M,A} <: AbstractMCMC.AbstractSampler
    "Initial [`AbstractMCMCKernel`](@ref)."
    initial_kernel::K
    "Initial [`AbstractMetric`](@ref)."
    initial_metric::M
    "Initial [`AbstractAdaptor`](@ref)."
    initial_adaptor::A
end
HMCSampler(kernel, metric) = HMCSampler(kernel, metric, Adaptation.NoAdaptation())

########
# NUTS #
########

function NUTS_kernel(integrator)
    return HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
end    

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
struct NUTS <: AdaptiveHamiltonian
    n_adapts::Int     # number of samples with adaption for ϵ
    TAP::Float64      # target accept rate
    max_depth::Int    # maximum tree depth
    Δ_max::Float64    # maximum error
    ϵ::Float64        # (initial) step size
    metric
    integrator
    kernel
    adaptor
end

function NUTS(
    n_adapts::Int,
    TAP::Float64; # Target Acceptance Probability 
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metric=nothing,
    integrator=Leapfrog)   
    function adaptor(metric, integrator)
        return StanHMCAdaptor(MassMatrixAdaptor(metric),
                              StepSizeAdaptor(TAP, integrator))
    end                          
    NUTS(n_adapts, TAP, max_depth, Δ_max, init_ϵ, metric, integrator, NUTS_kernel, adaptor)
end 

export NUTS 
#######
# HMC #
#######

function HMC_kernel(n_leapfrog)
    function kernel(integrator)
        return HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_leapfrog)))
    end
    return kernel    
end

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
struct HMC <: StaticHamiltonian
    ϵ::Float64 # leapfrog step size
    n_leapfrog::Int # leapfrog step number
    metric
    integrator
    kernel
end

function HMC(
    ϵ::Float64,
    n_leapfrog::Int;
    metric=nothing,
    integrator=Leapfrog)
    kernel = HMC_kernel(n_leapfrog)
    adaptor = Adaptation.NoAdaptation()
    return HMC(ϵ, n_leapfrog, metric, integrator, kernel, adaptor)
end

export HMC
#########
# HMCDA #
#########

function HMCDA_kernel(λ)
    function kernel(integrator)
        return HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(λ)))
    end
    return kernel    
end

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
struct HMCDA <: AdaptiveHamiltonian
    n_adapts    ::  Int         # number of samples with adaption for ϵ
    TAP         ::  Float64     # target accept rate
    λ           ::  Float64     # target leapfrog length
    ϵ           ::  Float64     # (initial) step size
    metric
    integrator
    kernel
end

function HMCDA(
    n_adapts::Int,
    TAP::Float64,
    λ::Float64;
    ϵ::Float64=0.0,
    metric=nothing,
    integrator=Leapfrog)
    kernel = HMCDA_kernel(λ)
    function adaptor(metric, integrator)
        return StanHMCAdaptor(MassMatrixAdaptor(metric),
                              StepSizeAdaptor(TAP, integrator))
    end    
    return HMCDA(n_adapts, TAP, λ, ϵ, metric, integrator, kernel, adaptor)
end

export HMCDA