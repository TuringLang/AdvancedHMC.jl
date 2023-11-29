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

"""
    $(TYPEDSIGNATURES)

A convenient wrapper around `AbstractMCMC.sample` avoiding explicit construction of [`HMCSampler`](@ref).
"""

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractHMCSampler,
    N::Integer;
    n_adapts::Int = min(div(N, 10), 1_000),
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...,
)
    if callback === nothing
        callback = HMCProgressCallback(N, progress = progress, verbose = verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        n_adapts = n_adapts,
        progress = progress,
        verbose = verbose,
        callback = callback,
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
    n_adapts::Int = min(div(N, 10), 1_000),
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...,
)

    if callback === nothing
        callback = HMCProgressCallback(N, progress = progress, verbose = verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        parallel,
        N,
        nchains;
        n_adapts = n_adapts,
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...,
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::AbstractHMCSampler;
    initial_params = nothing,
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
    n_adapts::Int = 0,
    kwargs...,
)
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
    tstat = merge(tstat, (is_adapt = isadapted,))

    # Compute next transition and state.
    newstate = HMCState(i, t, h.metric, κ, adaptor)

    # Return `Transition` with additional stats added.
    return Transition(t.z, tstat), newstate
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
    "`Progress` meter from ProgressMeters.jl."
    pm::P
    "Specifies whether or not to use display a progress bar."
    progress::Bool
    "If `progress` is not specified and this is `true` some information will be logged upon completion of adaptation."
    verbose::Bool
    "Number of divergent transitions fo far."
    num_divergent_transitions::Ref{Int}
    num_divergent_transitions_during_adaption::Ref{Int}
end

function HMCProgressCallback(n_samples; progress = true, verbose = false)
    pm =
        progress ? ProgressMeter.Progress(n_samples, desc = "Sampling", barlen = 31) :
        nothing
    HMCProgressCallback(pm, progress, verbose, Ref(0), Ref(0))
end

function (cb::HMCProgressCallback)(rng, model, spl, t, state, i; nadapts = 0, kwargs...)
    progress = cb.progress
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
    if progress
        percentage_divergent_transitions = cb.num_divergent_transitions[] / i
        percentage_divergent_transitions_during_adaption =
            cb.num_divergent_transitions_during_adaption[] / i
        if percentage_divergent_transitions > 0.25
            @warn "The level of numerical errors is high. Please check the model carefully." maxlog =
                3
        end
        # Do include current iteration and mass matrix
        pm_next!(
            pm,
            (
                iterations = i,
                ratio_divergent_transitions = round(
                    percentage_divergent_transitions;
                    digits = 2,
                ),
                ratio_divergent_transitions_during_adaption = round(
                    percentage_divergent_transitions_during_adaption;
                    digits = 2,
                ),
                tstat...,
                mass_matrix = metric,
            ),
        )
        # Report finish of adapation
    elseif verbose && isadapted && i == nadapts
        @info "Finished $nadapts adapation steps" adaptor κ.τ.integrator metric
    end
end

#############
### Utils ###
#############
function make_initial_params(
    rng::AbstractRNG,
    spl::AbstractHMCSampler,
    logdensity,
    initial_params,
)
    T = sampler_eltype(spl)
    if initial_params == nothing
        d = LogDensityProblems.dimension(logdensity)
        initial_params = randn(rng, d)
    end
    return T.(initial_params)
end

#########

function make_step_size(
    rng::Random.AbstractRNG,
    spl::HMCSampler,
    hamiltonian::Hamiltonian,
    initial_params,
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
    T::Type,
    hamiltonian::Hamiltonian,
    initial_params,
)
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
    T::Type,
    hamiltonian::Hamiltonian,
    initial_params,
)
    ϵ = find_good_stepsize(rng, hamiltonian, initial_params)
    @info string("Found initial step size ", ϵ)
    return T(ϵ)
end

make_integrator(spl::HMCSampler, ϵ::Real) = spl.κ.τ.integrator
make_integrator(spl::AbstractHMCSampler, ϵ::Real) = make_integrator(spl.integrator, ϵ)
make_integrator(i::AbstractIntegrator, ϵ::Real) = i
make_integrator(i::Symbol, ϵ::Real) = make_integrator(Val(i), ϵ)
make_integrator(@nospecialize(i), ::Real) = error("Integrator $i not supported.")
make_integrator(i::Val{:leapfrog}, ϵ::Real) = Leapfrog(ϵ)
make_integrator(i::Val{:jitteredleapfrog}, ϵ::T) where {T<:Real} =
    JitteredLeapfrog(ϵ, T(0.1ϵ))
make_integrator(i::Val{:temperedleapfrog}, ϵ::T) where {T<:Real} = TemperedLeapfrog(ϵ, T(1))

#########

make_metric(@nospecialize(i), T::Type, d::Int) = error("Metric $(typeof(i)) not supported.")
make_metric(i::Symbol, T::Type, d::Int) = make_metric(Val(i), T, d)
make_metric(i::AbstractMetric, T::Type, d::Int) = i
make_metric(i::Val{:diagonal}, T::Type, d::Int) = DiagEuclideanMetric(T, d)
make_metric(i::Val{:unit}, T::Type, d::Int) = UnitEuclideanMetric(T, d)
make_metric(i::Val{:dense}, T::Type, d::Int) = DenseEuclideanMetric(T, d)

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

function make_adaptor(
    spl::HMCSampler,
    metric::AbstractMetric,
    integrator::AbstractIntegrator,
)
    return spl.adaptor
end

#########

function make_kernel(spl::NUTS, integrator::AbstractIntegrator)
    return HMCKernel(
        Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(spl.max_depth, spl.Δ_max)),
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
