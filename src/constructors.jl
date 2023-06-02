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
    δ::Float64        # target accept rate
    max_depth::Int    # maximum tree depth
    Δ_max::Float64    # maximum error
    ϵ::Float64        # (initial) step size
    metric
    integrator
end

function NUTS(
    n_adapts::Int,
    δ::Float64,
    space::Symbol...;
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metric=nothing,
    integrator=Leapfrog,
)
    NUTS(n_adapts, δ, max_depth, Δ_max, init_ϵ, metric, integrator)
end

function NUTS(ϵ::Float64, TAP::Float64)
    metric =  DiagEuclideanMetric(d)
    integrator = Leapfrog(ϵ)
    kernel = NUTS{MultinomialTS, GeneralisedNoUTurn}
    adaptor(metric, integrator) = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(TAP, integrator))
    return HMCSampler(kernel, metric, adaptor)
end    
