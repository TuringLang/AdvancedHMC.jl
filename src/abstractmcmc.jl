
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

################
# No glue code #
################
function AbstractMCMC.sample(
    model::DynamicPPL.Model, 
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer;
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...,
)
    return AbstractMCMC.sample(
        Random.GLOBAL_RNG,
        model,
        sampler,
        N;
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...,
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer;
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
        param_names = vsyms,
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...,
    )
end
###

"""
    $(TYPEDSIGNATURES)

A convenient wrapper around `AbstractMCMC.sample` avoiding explicit construction of [`HMCSampler`](@ref).
"""
function AbstractMCMC.sample(
    model::LogDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    N::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(
        Random.GLOBAL_RNG,
        model,
        kernel,
        metric,
        adaptor,
        N;
        kwargs...,
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::LogDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    N::Integer;
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...,
)
    sampler = HMCSampler(kernel, metric, adaptor)
    if callback === nothing
        callback = HMCProgressCallback(N, progress = progress, verbose = verbose)
        progress = false # don't use AMCMC's progress-funtionality
    end

    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...,
    )
end

function AbstractMCMC.sample(
    model::LogDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    parallel::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    kwargs...,
)
    return AbstractMCMC.sample(
        Random.GLOBAL_RNG,
        model,
        kernel,
        metric,
        adaptor,
        N,
        nchains;
        kwargs...,
    )
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::LogDensityModel,
    kernel::AbstractMCMCKernel,
    metric::AbstractMetric,
    adaptor::AbstractAdaptor,
    parallel::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    progress = true,
    verbose = false,
    callback = nothing,
    kwargs...,
)
    sampler = HMCSampler(kernel, metric, adaptor)
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
        progress = progress,
        verbose = verbose,
        callback = callback,
        kwargs...,
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.model,
    spl::HMCSampler,
    vi # what type is this?;
    init_params = nothing,
    kwargs...,
)   
    # unpack model
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)
    # make model from Turing output
    ℓ = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(vi, model, ctxt))
    model = AbstractMCMC.LogDensityModel(ℓ)

    # We will need to implement this but it is going to be
    # Interesting how to plug the transforms along the sampling
    # processes
    #vi_t = Turing.link!!(vi, model)
    dists = _get_dists(vi)
    dist_lengths = [length(dist) for dist in dists]
    vsyms = _name_variables(vi, dist_lengths)
    d = LogDensityProblems.dimension(ℓ)

    # Define metric
    if spl.metric == nothing
        metric = DiagEuclideanMetric(d)
    else
        metric = spl.metric    
    end

    # Construct the hamiltonian using the initial metric
    hamiltonian = Hamiltonian(metric, model)

    # Find good eps if not provided one
    if iszero(spl.alg.ϵ)
        # Extract parameters.
        theta = vi[spl]
        ϵ = AHMC.find_good_stepsize(rng, hamiltonian, theta)
        @info "Found initial step size" ϵ
    else
        ϵ = spl.alg.ϵ
    end

    integrator = spl.integrator(ϵ)
    κ = spl.kernel(integrator)
    adaptor = spl.adaptor(metric, integrator)
    spl = HMCSampler(kernel, metric, adaptor)

    if init_params === nothing
        init_params = randn(rng, size(metric, 1))
    end

    # Get an initial sample.
    h, t = AdvancedHMC.sample_init(rng, hamiltonian, init_params)

    # Compute next transition and state.
    state = HMCState(0, t, h.metric, κ, adaptor)

    # Take actual first step.
    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::LogDensityModel,
    spl::HMCSampler,
    state::HMCState;
    nadapts::Int = 0,
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
    h, κ, isadapted = adapt!(h, κ, adaptor, i, nadapts, t.z.θ, tstat.acceptance_rate)
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
