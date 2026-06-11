############
# Stepsize #
############

struct PolynomialStepsize{T<:Real}
    "Constant scale factor of the step size."
    a::T
    "Constant offset of the step size."
    b::T
    "Decay rate of step size in (0.5, 1]."
    γ::T

    function PolynomialStepsize{T}(a::T, b::T, γ::T) where {T}
        0.5 < γ ≤ 1 || error("the decay rate `γ` has to be in (0.5, 1]")
        return new{T}(a, b, γ)
    end
end
Base.eltype(p::PolynomialStepsize{T}) where {T} = T

"""
    PolynomialStepsize(a[, b=0, γ=0.55])

Create a polynomially decaying stepsize function.

At iteration `t`, the step size is
```math
a / (b + t)^γ.
```
"""
function PolynomialStepsize(a::T, b::T, γ::T) where {T<:Real}
    return PolynomialStepsize{T}(a, b, γ)
end
function PolynomialStepsize(a::Real, b::Real=0, γ::Real=0.55)
    return PolynomialStepsize(promote(a, b, γ)...)
end

(f::PolynomialStepsize)(t::Int) = f.a / (t + f.b)^f.γ

############
# Sampler  #
############

"""
    SGLD(stepsize; metric = :unit)

Stochastic Gradient Langevin Dynamics sampler using the Welling & Teh (2011)
update with a decreasing step size and optional constant preconditioning.

The model's gradient must be an unbiased stochastic gradient estimate of the
full log posterior. If minibatching is used, the model must include the prior
term and the `N / n` likelihood scaling. With a deterministic full gradient,
this reduces to decreasing-step unadjusted Langevin dynamics; it is not exact at
fixed step size and has no Metropolis-Hastings correction.

AdvancedHMC does not compute the sampling-threshold diagnostic from the paper.
Discard pre-threshold draws downstream, and use post-burn-in step-size weighted
posterior estimates, e.g.

```julia
sum(t.stat.step_size * f(t.z.θ) for t in samples) /
sum(t.stat.step_size for t in samples)
```

# Fields

$(FIELDS)

# Notes

For more information, please view the following paper:
 - Max Welling & Yee Whye Teh (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. In: Proceedings of the 28th International Conference on Machine Learning (pp. 681–688).
"""
struct SGLD{S,M<:Union{Symbol,AbstractMetric}} <: AbstractHMCSampler
    "Polynomial step size function."
    stepsize::S
    "Fixed preconditioning metric. `Symbol` means it is automatically initialised."
    metric::M
end

function SGLD(stepsize; metric=:unit)
    return SGLD(stepsize, metric)
end

sampler_eltype(sampler::SGLD) = determine_sampler_eltype(sampler.stepsize, sampler.metric)

#########
# State #
#########

struct SGLDState{TTrans<:Transition,TMetric<:AbstractMetric}
    "Index of current iteration."
    i::Int
    "Current [`Transition`](@ref)."
    transition::TTrans
    "Fixed [`AbstractMetric`](@ref) preconditioner."
    metric::TMetric
end
getmetric(state::SGLDState) = state.metric

function AbstractMCMC.getparams(state::SGLDState)
    return state.transition.z.θ
end

function AbstractMCMC.setparams!!(
    model::AbstractMCMC.LogDensityModel, state::SGLDState, params
)
    hamiltonian = Hamiltonian(state.metric, model)
    return Setfield.@set state.transition.z = phasepoint(
        hamiltonian, params, zero(params)
    )
end

###################
# SGLD transition #
###################

_sgld_preconditioned_gradient(::UnitEuclideanMetric, grad) = grad
_sgld_preconditioned_noise(::UnitEuclideanMetric, noise) = noise

function _sgld_preconditioned_gradient(metric::DiagEuclideanMetric, grad)
    return _sgld_scale(metric.M⁻¹, grad)
end
function _sgld_preconditioned_noise(metric::DiagEuclideanMetric, noise)
    return _sgld_scale(metric.sqrtM⁻¹, noise)
end

_sgld_preconditioned_gradient(metric::DenseEuclideanMetric, grad) = metric.M⁻¹ * grad
_sgld_preconditioned_noise(metric::DenseEuclideanMetric, noise) =
    metric.cholM⁻¹' * noise

_sgld_scale(scale::AbstractVector, x::AbstractVector) = scale .* x
_sgld_scale(scale::AbstractVector, x::AbstractMatrix) = reshape(scale, :, 1) .* x
_sgld_scale(scale::AbstractMatrix, x::AbstractVecOrMat) = scale .* x

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::SGLD;
    initial_params=nothing,
    kwargs...,
)
    # Unpack model
    logdensity = model.logdensity

    # Define metric
    metric = make_metric(spl, logdensity)

    # Construct the hamiltonian using the initial metric
    hamiltonian = Hamiltonian(metric, model)

    # Compute initial sample and state.
    initial_params = make_initial_params(rng, spl, logdensity, initial_params)
    hamiltonian = resize(hamiltonian, initial_params)
    metric = hamiltonian.metric

    # SGLD has no momentum; keep this zero to avoid random draws and spurious kinetic energy.
    t = Transition(phasepoint(hamiltonian, initial_params, zero(initial_params)), NamedTuple())
    state = SGLDState(0, t, metric)

    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::SGLD,
    state::SGLDState;
    n_adapts::Int=0,
    kwargs...,
)
    if haskey(kwargs, :nadapts)
        throw(
            ArgumentError(
                "keyword argument `nadapts` is unsupported. Please use `n_adapts` to specify the number of adaptation steps.",
            ),
        )
    end

    i = state.i + 1
    t_old = state.transition
    metric = state.metric

    # Reconstruct hamiltonian.
    h = Hamiltonian(metric, model)

    # Compute gradient of log density.
    logdensity_and_gradient = Base.Fix1(
        LogDensityProblems.logdensity_and_gradient, model.logdensity
    )
    θ = copy(t_old.z.θ)
    _, grad = logdensity_and_gradient(θ)

    stepsize = spl.stepsize(i)
    noise = randn(rng, eltype(θ), size(θ)...)
    θ .+= (stepsize / 2) .* _sgld_preconditioned_gradient(metric, grad) .+
          sqrt(stepsize) .* _sgld_preconditioned_noise(metric, noise)

    # Make new transition.
    z = phasepoint(h, θ, zero(θ))
    tstat = (
        n_steps=1,
        step_size=stepsize,
        is_accept=true,
        acceptance_rate=one(eltype(θ)),
        log_density=z.ℓπ.value,
        numerical_error=!isfinite(z),
        is_adapt=false,
    )
    t = Transition(z, tstat)

    # Compute next sample and state.
    newstate = SGLDState(i, t, metric)

    return t, newstate
end
