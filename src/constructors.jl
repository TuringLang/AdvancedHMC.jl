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

struct NUTS_kernel{TS,TC} end

"""
$(SIGNATURES)

Convenient constructor for the no-U-turn sampler (NUTS).
This falls back to `HMCKernel(Trajectory{TS}(int, TC(args...; kwargs...)))` where

- `TS<:Union{MultinomialTS, SliceTS}` is the type for trajectory sampler
- `TC<:Union{ClassicNoUTurn, GeneralisedNoUTurn, StrictGeneralisedNoUTurn}` is the type for termination criterion.

See [`ClassicNoUTurn`](@ref), [`GeneralisedNoUTurn`](@ref) and [`StrictGeneralisedNoUTurn`](@ref) for details in parameters.
"""
NUTS_kernel{TS,TC}(int::AbstractIntegrator, args...; kwargs...) where {TS,TC} =
    HMCKernel(Trajectory{TS}(int, TC(args...; kwargs...)))
NUTS_kernel(int::AbstractIntegrator, args...; kwargs...) =
    HMCKernel(Trajectory{MultinomialTS}(int, GeneralisedNoUTurn(args...; kwargs...)))
NUTS_kernel(ϵ::AbstractScalarOrVec{<:Real}) =
    HMCKernel(Trajectory{MultinomialTS}(Leapfrog(ϵ), GeneralisedNoUTurn()))

export NUTS

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
    TAP::Float64        # target accept rate
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
    integrator=Leapfrog,
    kernel = NUTS_kernel{MultinomialTS, GeneralisedNoUTurn}
)   
    function adaptor(metric, integrator)
        return StanHMCAdaptor(MassMatrixAdaptor(metric),
                              StepSizeAdaptor(TAP, integrator))
    end                          
    NUTS(n_adapts, TAP, max_depth, Δ_max, init_ϵ, metric, integrator, kernel, adaptor)
end 
