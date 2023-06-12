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
    TV<:AbstractVarInfo,
}
    "Current Var Info"
    vi::TV
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
    "Current [`Hamiltonian`](@ref)."
    hamiltonian::Hamiltonian
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AbstractMCMC.AbstractSampler;
    init_params = nothing,
    kwargs...,
)   
    # Unpack model
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)
    vi_t = DynamicPPL.link!!(vi, model)
    logdensityfunction = DynamicPPL.LogDensityFunction(vi_t, model, ctxt)
    logdensityproblem = LogDensityProblemsAD.ADgradient(logdensityfunction)
    logdensitymodel = AbstractMCMC.LogDensityModel(logdensityproblem)

    # Define metric
    if spl.metric == nothing
        d = LogDensityProblems.dimension(logdensityproblem)
        metric = DiagEuclideanMetric(d)
    else
        metric = spl.metric    
    end

    # Construct the hamiltonian using the initial metric
    hamiltonian = Hamiltonian(metric, logdensitymodel)

    # Define integration algorithm
    if spl.integrator == nothing
        # Find good eps if not provided one
        if iszero(spl.alg.ϵ)
            # Extract parameters.
            theta = vi_t[spl]
            ϵ = find_good_stepsize(rng, hamiltonian, theta)
            println(string("Found initial step size ", ϵ))
        else
            ϵ = spl.alg.ϵ
        end
        integrator = Leapfrog(ϵ)
    else
        integrator = spl.integrator
    end

    # Make kernel
    kernel = make_kernel(spl.alg, integrator)

    # Make adaptor
    if spl.adaptor == nothing
        if typeof(spl.alg) <: AdvancedHMC.AdaptiveHamiltonian
            adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
                                     StepSizeAdaptor(spl.alg.TAP, integrator))
            n_adapts = spl.alg.n_adapts
        else
            adaptor = NoAdaptation()
            n_adapts = 0
        end
    else
        adaptor = spl.adaptor
        n_adapts = kwargs[:n_adapts]
    end        

    if init_params == nothing
        init_params = vi_t[DynamicPPL.SampleFromPrior()]
    else
        init_params = init_params
        # We have to think of a way of transforming the initial parameters 
        # init_params = DynamicPPL.link!!()
    end

    # Get an initial sample.
    h, t = AdvancedHMC.sample_init(rng, hamiltonian, init_params)

    # Compute next transition and state.
    state = HMCState(vi, 0, t, h.metric, kernel, adaptor, hamiltonian)
    # Take actual first step.
    return AbstractMCMC.step(
        rng,
        model,
        spl,
        state;
        n_adapts=n_adapts,
        kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AbstractMCMC.AbstractSampler,
    state::HMCState;
    nadapts::Int=0,
    kwargs...,
)   
    # Get step size
    @debug "current ϵ" getstepsize(spl, state)

    # Compute transition.
    i = state.i + 1
    t_old = state.transition
    adaptor = state.adaptor
    κ = state.κ
    metric = state.metric
    h = state.hamiltonian
    vi = state.vi
    vi_t = DynamicPPL.link!!(vi, model)

    # Make new transition.
    t = transition(rng, h, κ, t_old.z)

    # Adapt h and spl.
    tstat = stat(t)
    h, κ, isadapted = adapt!(h, κ, adaptor, i, nadapts, t.z.θ, tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt = isadapted,))

    # Convert variables back
    vii_t = DynamicPPL.unflatten(vi_t, t.z.θ)  
    vii = DynamicPPL.invlink!!(vii_t, model)
    θ = vii[spl]
    zz = phasepoint(rng, θ, h)

    # Compute next transition and state.
    newstate = HMCState(vii, i, t, h.metric, κ, adaptor, h)

    # Return `Transition` with additional stats added.
    return Transition(zz, tstat), newstate
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
