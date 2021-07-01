struct HMCSampler{K, M, A} <: AbstractMCMC.AbstractSampler
    initial_kernel::K
    initial_metric::M
    initial_adaptor::A
end
HMCSampler(kernel, metric) = HMCSampler(kernel, metric, Adaptation.NoAdaptation())

struct DifferentiableDensityModel{Tlogπ, T∂logπ∂θ} <: AbstractMCMC.AbstractModel
    ℓπ::Tlogπ
    ∂ℓπ∂θ::T∂logπ∂θ
end

struct HMCState{
    TTrans<:Transition,
    TMetric<:AbstractMetric,
    TKernel<:AbstractMCMCKernel,
    TAdapt<:Adaptation.AbstractAdaptor
}
    i::Int
    transition::TTrans
    metric::TMetric
    κ::TKernel
    adaptor::TAdapt
end

function AbstractMCMC.sample(
    model::DifferentiableDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    N::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, kernel, metric, adaptor, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DifferentiableDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    N::Integer;
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...
)
    sampler = HMCSampler(kernel, metric, adaptor)
    if callback === nothing
        callback = HMCProgressCallback(N, progress = progress, verbose = verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng, model, sampler, N;
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...
    )
end

function AbstractMCMC.sample(
    model::DifferentiableDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    nchains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(
        Random.GLOBAL_RNG, model, kernel, metric, adaptor, N, nchains;
        kwargs...
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DifferentiableDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    nchains::Integer;
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...
)
    sampler = HMCSampler(kernel, metric, adaptor)
    if callback === nothing
        callback = HMCProgressCallback(N, progress = progress, verbose = verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng, model, sampler, parallel, N, nchains;
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DifferentiableDensityModel,
    spl::HMCSampler;
    init_params = nothing,
    kwargs...
)
    metric = spl.initial_metric
    κ = spl.initial_kernel
    adaptor = spl.initial_adaptor

    if init_params === nothing
        init_params = randn(size(metric, 1))
    end

    # Construct the hamiltonian using the initial metric
    hamiltonian = Hamiltonian(metric, model.ℓπ, model.∂ℓπ∂θ)

    # Get an initial sample.
    h, t = AdvancedHMC.sample_init(rng, hamiltonian, init_params)

    # Compute next transition and state.
    state = HMCState(0, t, h.metric, κ, adaptor)

    # Take actual first step.
    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DifferentiableDensityModel,
    spl::HMCSampler,
    state::HMCState;
    nadapts::Int = 0,
    kwargs...
)
    # Get step size
    @debug "current ϵ" getstepsize(spl, state)

    # Compute transition.
    i = state.i + 1
    t_old = state.transition
    adaptor = state.adaptor
    κ = state.κ
    metric = state.metric

    # Reconstruct hamiltonian.
    h = Hamiltonian(metric, model.ℓπ, model.∂ℓπ∂θ)

    # Make new transition.
    t = transition(rng, h, κ, t_old.z)

    # Adapt h and spl.
    tstat = stat(t)
    h, κ, isadapted = adapt!(h, κ, adaptor, i, nadapts, t.z.θ, tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt=isadapted,))

    # Compute next transition and state.
    newstate = HMCState(i, t, h.metric, κ, adaptor)

    # Return `Transition` with additional stats added.
    return Transition(t.z, tstat), newstate
end


################
### Callback ###
################
struct HMCProgressCallback{P}
    pm::P
    progress::Bool
    verbose::Bool
end

function HMCProgressCallback(n_samples; progress=true, verbose=false)
    pm = progress ? ProgressMeter.Progress(n_samples, desc="Sampling", barlen=31) : nothing
    HMCProgressCallback(pm, progress, verbose)
end

function (cb::HMCProgressCallback)(
    rng, model, spl, t, state, i;
    nadapts = 0,
    kwargs...
)
    progress = cb.progress
    verbose = cb.verbose
    pm = cb.pm
    
    metric = state.metric
    adaptor = state.adaptor
    κ = state.κ
    tstat = t.stat
    isadapted = tstat.is_adapt

    # Update progress meter
    if progress
        # Do include current iteration and mass matrix
        pm_next!(
            pm,
            (iterations=i, tstat..., mass_matrix=metric)
        )
        # Report finish of adapation
    elseif verbose && isadapted && i == nadapts
        @info "Finished $nadapts adapation steps" adaptor κ.τ.integrator metric
    end
end
