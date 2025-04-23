"""
    HMCState

Represents the state of a [`HMCSampler`](@ref).

# Fields

$(FIELDS)

"""
struct HMCState{
    TTrans<:Transition,
    TMetric<:AbstractMetric,
    TKernel<:AbstractMCMCKernel,
    TAdapt<:Adaptation.AbstractAdaptor,
}
    "Index of current iteration."
    i::Int
    "Current [`Transition`](@ref)."
    transition::TTrans
    "Current [`AbstractMetric`](@ref), possibly adapted."
    metric::TMetric
    "Current [`AbstractMCMCKernel`](@ref)."
    κ::TKernel
    "Current [`AbstractAdaptor`](@ref)."
    adaptor::TAdapt
end

getadaptor(state::HMCState) = state.adaptor
getmetric(state::HMCState) = state.metric
getintegrator(state::HMCState) = state.κ.τ.integrator

function AbstractMCMC.getparams(state::HMCState)
    return state.transition.z.θ
end

function AbstractMCMC.setparams!!(
    model::AbstractMCMC.LogDensityModel, state::HMCState, params
)
    hamiltonian = AdvancedHMC.Hamiltonian(state.metric, model)
    return Setfield.@set state.transition.z = AdvancedHMC.phasepoint(
        hamiltonian, params, state.transition.z.r; ℓκ=state.transition.z.ℓκ
    )
end

"""
    $(TYPEDSIGNATURES)

A convenient wrapper around `AbstractMCMC.sample` avoiding explicit construction of [`HMCSampler`](@ref).
"""

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractHMCSampler,
    N::Integer;
    n_adapts::Int=min(div(N, 10), 1_000),
    progress=true,
    verbose=false,
    callback=nothing,
    kwargs...,
)
    if haskey(kwargs, :nadapts)
        throw(
            ArgumentError(
                "keyword argument `nadapts` is unsupported. Please use `n_adapts` to specify the number of adaptation steps.",
            ),
        )
    end

    if callback === nothing
        callback = HMCProgressCallback(N; progress=progress, verbose=verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        n_adapts=n_adapts,
        progress=progress,
        verbose=verbose,
        callback=callback,
        kwargs...,
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractHMCSampler,
    parallel::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    n_adapts::Int=min(div(N, 10), 1_000),
    progress=true,
    verbose=false,
    callback=nothing,
    kwargs...,
)
    if haskey(kwargs, :nadapts)
        throw(
            ArgumentError(
                "keyword argument `nadapts` is unsupported. Please use `n_adapts` to specify the number of adaptation steps.",
            ),
        )
    end

    if callback === nothing
        callback = HMCProgressCallback(N; progress=progress, verbose=verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        parallel,
        N,
        nchains;
        n_adapts=n_adapts,
        progress=progress,
        verbose=verbose,
        callback=callback,
        kwargs...,
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::AbstractHMCSampler;
    initial_params=nothing,
    kwargs...,
)
    # Unpack model
    logdensity = model.logdensity

    # Define metric
    metric = make_metric(spl, logdensity)

    # Construct the hamiltonian using the initial metric
    hamiltonian = Hamiltonian(metric, model)

    # Define integration algorithm
    # Find good eps if not provided one
    initial_params = make_initial_params(rng, spl, logdensity, initial_params)
    ϵ = make_step_size(rng, spl, hamiltonian, initial_params)
    integrator = make_integrator(spl, ϵ)

    # Make kernel
    κ = make_kernel(spl, integrator)

    # Make adaptor
    adaptor = make_adaptor(spl, metric, integrator)

    # Get an initial sample.
    h, t = AdvancedHMC.sample_init(rng, hamiltonian, initial_params)

    # Compute next transition and state.
    state = HMCState(0, t, metric, κ, adaptor)
    # Take actual first step.
    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::AbstractHMCSampler,
    state::HMCState;
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

    # Compute transition.
    i = state.i + 1
    t_old = state.transition
    adaptor = state.adaptor
    κ = state.κ
    metric = state.metric

    # Reconstruct hamiltonian.
    h = Hamiltonian(metric, model)

    # Make new transition.
    t = transition(rng, h, κ, t_old.z)

    # Adapt h and spl.
    tstat = stat(t)
    h, κ, isadapted = adapt!(h, κ, adaptor, i, n_adapts, t.z.θ, tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt=isadapted,))

    # Compute next transition and state.
    newstate = HMCState(i, t, h.metric, κ, adaptor)

    # Return `Transition` with additional stats added.
    return Transition(t.z, tstat), newstate
end

struct SGHMCState{T<:AbstractVector{<:Real}}
    "Index of current iteration."
    i
    "Current [`Transition`](@ref)."
    transition
    "Current [`AbstractMetric`](@ref), possibly adapted."
    metric
    "Current [`AbstractMCMCKernel`](@ref)."
    κ
    "Current [`AbstractAdaptor`](@ref)."
    adaptor
    velocity::T
end
getadaptor(state::SGHMCState) = state.adaptor
getmetric(state::SGHMCState) = state.metric
getintegrator(state::SGHMCState) = state.κ.τ.integrator

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::SGHMC;
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
    ϵ = make_step_size(rng, spl, hamiltonian, initial_params)
    integrator = make_integrator(spl, ϵ)

    # Make kernel
    κ = make_kernel(spl, integrator)

    # Make adaptor
    adaptor = make_adaptor(spl, metric, integrator)

    # Get an initial sample.
    h, t = AdvancedHMC.sample_init(rng, hamiltonian, initial_params)

    state = SGHMCState(0, t, metric, κ, adaptor, initial_params, zero(initial_params))

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
    i = state.i + 1
    t_old = state.transition
    adaptor = state.adaptor
    κ = state.κ
    metric = state.metric

    # Reconstruct hamiltonian.
    h = Hamiltonian(metric, model)

    # Compute gradient of log density.
    logdensity_and_gradient = Base.Fix1(
        LogDensityProblems.logdensity_and_gradient, model.logdensity
    )
    θ = t_old.z.θ
    grad = last(logdensity_and_gradient(θ))

    # Update latent variables and velocity according to
    # equation (15) of Chen et al. (2014)
    v = state.velocity
    θ .+= v
    η = spl.learning_rate
    α = spl.momentum_decay
    newv = (1 - α) .* v .+ η .* grad .+ sqrt(2 * η * α) .* randn(rng, eltype(v), length(v))

    # Adapt h and spl.
    tstat = stat(t)
    h, κ, isadapted = adapt!(h, κ, adaptor, i, n_adapts, θ, tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt=isadapted,))

    # Make new transition.
    t = transition(rng, h, κ, t_old.z)

    # Compute next sample and state.
    sample = Transition(t.z, tstat)
    newstate = SGHMCState(i, t, h.metric, κ, adaptor, newv)

    return sample, newstate
end

################
### Callback ###
################
"""
    HMCProgressCallback

A callback to be used with AbstractMCMC.jl's interface, replicating the
logging behavior of the non-AbstractMCMC [`sample`](@ref).

# Fields
$(FIELDS)
"""
struct HMCProgressCallback{P}
    "`Progress` meter from ProgressMeters.jl, or `nothing`."
    pm::P
    "If `pm === nothing` and this is `true` some information will be logged upon completion of adaptation."
    verbose::Bool
    "Number of divergent transitions."
    num_divergent_transitions::Base.RefValue{Int}
    num_divergent_transitions_during_adaption::Base.RefValue{Int}
end

function HMCProgressCallback(n_samples; progress=true, verbose=false)
    pm = progress ? ProgressMeter.Progress(n_samples; desc="Sampling", barlen=31) : nothing
    return HMCProgressCallback(pm, verbose, Ref(0), Ref(0))
end

function (cb::HMCProgressCallback)(rng, model, spl, t, state, i; n_adapts::Int=0, kwargs...)
    verbose = cb.verbose
    pm = cb.pm

    metric = state.metric
    adaptor = state.adaptor
    κ = state.κ
    tstat = t.stat
    isadapted = tstat.is_adapt
    if isadapted
        cb.num_divergent_transitions_during_adaption[] += tstat.numerical_error
    else
        cb.num_divergent_transitions[] += tstat.numerical_error
    end

    # Update progress meter
    if pm !== nothing
        percentage_divergent_transitions =
            cb.num_divergent_transitions[] / max(i - n_adapts, 1)
        percentage_divergent_transitions_during_adaption =
            cb.num_divergent_transitions_during_adaption[] / min(i, n_adapts)
        if percentage_divergent_transitions > 0.25
            @warn "The level of numerical errors is high. Please check the model carefully." maxlog =
                3
        end
        # Do include current iteration and mass matrix
        pm_next!(
            pm,
            (
                iterations=i,
                ratio_divergent_transitions=round(
                    percentage_divergent_transitions; digits=2
                ),
                ratio_divergent_transitions_during_adaption=round(
                    percentage_divergent_transitions_during_adaption; digits=2
                ),
                tstat...,
                mass_matrix=metric,
            ),
        )
        # Report finish of adapation
    elseif verbose && isadapted && i == n_adapts
        @info "Finished $(n_adapts) adapation steps" adaptor κ.τ.integrator metric
    end
end

#############
### Utils ###
#############
function make_initial_params(
    rng::AbstractRNG, spl::AbstractHMCSampler, logdensity, initial_params
)
    T = sampler_eltype(spl)
    if initial_params === nothing
        d = LogDensityProblems.dimension(logdensity)
        return randn(rng, T, d)
    else
        return T.(initial_params)
    end
end

#########

function make_step_size(
    rng::Random.AbstractRNG, spl::HMCSampler, hamiltonian::Hamiltonian, initial_params
)
    T = typeof(spl.κ.τ.integrator.ϵ)
    ϵ = make_step_size(rng, spl.κ.τ.integrator, T, hamiltonian, initial_params)
    return ϵ
end

function make_step_size(
    rng::Random.AbstractRNG,
    spl::AbstractHMCSampler,
    hamiltonian::Hamiltonian,
    initial_params,
)
    T = sampler_eltype(spl)
    return make_step_size(rng, spl.integrator, T, hamiltonian, initial_params)
end

function make_step_size(
    rng::Random.AbstractRNG,
    integrator::AbstractIntegrator,
    ::Type{T},
    hamiltonian::Hamiltonian,
    initial_params,
) where {T}
    if integrator.ϵ > 0
        ϵ = integrator.ϵ
    else
        ϵ = find_good_stepsize(rng, hamiltonian, initial_params)
        @info string("Found initial step size ", ϵ)
    end
    return T(ϵ)
end

function make_step_size(
    rng::Random.AbstractRNG,
    integrator::Symbol,
    ::Type{T},
    hamiltonian::Hamiltonian,
    initial_params,
) where {T}
    ϵ = find_good_stepsize(rng, hamiltonian, initial_params)
    @info string("Found initial step size ", ϵ)
    return T(ϵ)
end

make_integrator(spl::HMCSampler, ϵ::Real) = spl.κ.τ.integrator
make_integrator(spl::AbstractHMCSampler, ϵ::Real) = make_integrator(spl.integrator, ϵ)
make_integrator(i::AbstractIntegrator, ϵ::Real) = i
function make_integrator(i::Symbol, ϵ::Real)
    float_ϵ = AbstractFloat(ϵ)
    if i === :leapfrog
        return Leapfrog(float_ϵ)
    elseif i === :jitteredleapfrog
        return JitteredLeapfrog(float_ϵ, float_ϵ / 10)
    elseif i === :temperedleapfrog
        return TemperedLeapfrog(float_ϵ, oneunit(float_ϵ))
    else
        error("Integrator $i not supported.")
    end
end

#########

make_metric(i::AbstractMetric, ::Type, ::Int) = i
function make_metric(i::Symbol, ::Type{T}, d::Int) where {T}
    if i === :diagonal
        return DiagEuclideanMetric(T, d)
    elseif i === :unit
        return UnitEuclideanMetric(T, d)
    elseif i === :dense
        return DenseEuclideanMetric(T, d)
    else
        error("Metric $i not supported.")
    end
end

function make_metric(spl::AbstractHMCSampler, logdensity)
    d = LogDensityProblems.dimension(logdensity)
    T = sampler_eltype(spl)
    return make_metric(spl.metric, T, d)
end

#########

function make_adaptor(spl::NUTS, metric::AbstractMetric, integrator::AbstractIntegrator)
    return StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(spl.δ, integrator))
end

function make_adaptor(spl::HMCDA, metric::AbstractMetric, integrator::AbstractIntegrator)
    return StepSizeAdaptor(spl.δ, integrator)
end

function make_adaptor(spl::HMC, metric::AbstractMetric, integrator::AbstractIntegrator)
    return NoAdaptation()
end

function make_adaptor(spl::SGHMC, metric::AbstractMetric, integrator::AbstractIntegrator)
    return NoAdaptation()
end

function make_adaptor(
    spl::HMCSampler, metric::AbstractMetric, integrator::AbstractIntegrator
)
    return spl.adaptor
end

#########

function make_kernel(spl::NUTS, integrator::AbstractIntegrator)
    return HMCKernel(
        Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(spl.max_depth, spl.Δ_max))
    )
end

function make_kernel(spl::HMC, integrator::AbstractIntegrator)
    return HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(spl.n_leapfrog)))
end

function make_kernel(spl::HMCDA, integrator::AbstractIntegrator)
    return HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(spl.λ)))
end

function make_kernel(spl::HMCSampler, integrator::AbstractIntegrator)
    return spl.κ
end

function make_kernel(spl::SGHMC, integrator::AbstractIntegrator)
    return HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(spl.n_leapfrog)))
end
