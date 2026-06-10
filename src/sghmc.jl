#############
### SGHMC ###
#############
"""
    SGHMC(learning_rate::Real, momentum_decay::Real, n_steps::Int)

Stochastic Gradient Hamiltonian Monte Carlo sampler using the scalar Eq. 15
parameterization of Chen et al. (2014), with `β̂ = 0` and identity mass.
This sampler has no normal HMC trajectory, no momentum refreshment, and no
Metropolis-Hastings correction. The Eq. 15 velocity is stored in the sampler
state, not in `Transition.z.r`. Minibatch-gradient support is not yet exposed.

# Fields

$(FIELDS)

# Notes

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1402.4102)):

- Chen, Tianqi, Emily Fox, and Carlos Guestrin. "Stochastic gradient hamiltonian monte carlo." International conference on machine learning. PMLR, 2014.
"""
struct SGHMC{T<:Real} <: AbstractHMCSampler
    "Learning rate `η`."
    learning_rate::T
    "Momentum decay rate `α`."
    momentum_decay::T
    "Number of SGHMC inner integration steps."
    n_steps::Int
    function SGHMC{T}(learning_rate::T, momentum_decay::T, n_steps::Int) where {T<:Real}
        learning_rate >= 0 ||
            throw(ArgumentError("learning_rate must be nonnegative"))
        momentum_decay >= 0 ||
            throw(ArgumentError("momentum_decay must be nonnegative"))
        momentum_decay <= 1 || throw(
            ArgumentError(
                "momentum_decay must be <= 1 for this scalar Eq. 15 implementation",
            ),
        )
        n_steps >= 1 || throw(ArgumentError("n_steps must be positive"))
        return new{T}(learning_rate, momentum_decay, n_steps)
    end
end

function SGHMC(learning_rate, momentum_decay, n_steps::Int)
    T = determine_sampler_eltype(learning_rate, momentum_decay)
    return SGHMC{T}(T(learning_rate), T(momentum_decay), n_steps)
end

sampler_eltype(::SGHMC{T}) where {T} = T

struct SGHMCState{TTrans<:Transition,T<:AbstractVector{<:Real}}
    "Index of current iteration."
    i::Int
    "Current [`Transition`](@ref)."
    transition::TTrans
    "Current Eq. 15 velocity."
    velocity::T
end

_sghmc_hamiltonian(model::AbstractMCMC.LogDensityModel, θ) = Hamiltonian(
    UnitEuclideanMetric(eltype(θ), length(θ)), model
)

function AbstractMCMC.getparams(state::SGHMCState)
    return state.transition.z.θ
end

function AbstractMCMC.setparams!!(
    model::AbstractMCMC.LogDensityModel, state::SGHMCState, params
)
    hamiltonian = _sghmc_hamiltonian(model, params)
    return Setfield.@set state.transition.z = phasepoint(
        hamiltonian, params, zero(params)
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::SGHMC;
    initial_params=nothing,
    kwargs...,
)
    # Unpack model
    logdensity = model.logdensity

    # Compute initial sample and state.
    initial_params = make_initial_params(rng, spl, logdensity, initial_params)
    velocity = zero(initial_params)
    hamiltonian = _sghmc_hamiltonian(model, initial_params)
    t = Transition(phasepoint(hamiltonian, initial_params, velocity), NamedTuple())
    state = SGHMCState(0, t, velocity)

    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::SGHMC,
    state::SGHMCState;
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

    # Reconstruct hamiltonian.
    h = _sghmc_hamiltonian(model, t_old.z.θ)

    θ = copy(t_old.z.θ)
    v = copy(state.velocity)
    η, α = spl.learning_rate, spl.momentum_decay
    σ = sqrt(2 * η * α)

    θ .+= v
    value, grad = h.∂ℓπ∂θ(θ)
    v .= (1 - α) .* v .+ η .* grad .+ σ .* randn(rng, eltype(v), size(v)...)
    for _ in 2:(spl.n_steps)
        θ .+= v
        value, grad = h.∂ℓπ∂θ(θ)
        v .= (1 - α) .* v .+ η .* grad .+ σ .* randn(rng, eltype(v), size(v)...)
    end

    z = phasepoint(h, θ, zero(θ); ℓπ=DualValue(value, -grad))
    tstat = (
        n_steps=spl.n_steps,
        is_accept=true,
        acceptance_rate=one(eltype(v)),
        log_density=z.ℓπ.value,
        numerical_error=!all(isfinite, grad),
        is_adapt=false,
    )
    t = Transition(z, tstat)
    return t, SGHMCState(i, t, v)
end
