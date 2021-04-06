# Update of hamiltonian and proposal

reconstruct(h::Hamiltonian, ::AbstractAdaptor) = h
function reconstruct(
    h::Hamiltonian, adaptor::Union{MassMatrixAdaptor, NaiveHMCAdaptor, StanHMCAdaptor}
)
    metric = renew(h.metric, getM⁻¹(adaptor))
    return reconstruct(h, metric=metric)
end

reconstruct(τ::Trajectory, ::AbstractAdaptor) = τ
function reconstruct(
    τ::Trajectory, adaptor::Union{StepSizeAdaptor, NaiveHMCAdaptor, StanHMCAdaptor}
)
    # FIXME: this does not support change type of `ϵ` (e.g. Float to Vector)
    integrator = update_nom_step_size(τ.integrator, getϵ(adaptor))
    return reconstruct(τ, integrator=integrator)
end

reconstruct(κ::AbstractMCMCKernel, adaptor::AbstractAdaptor) = 
    reconstruct(κ, τ=reconstruct(κ.τ, adaptor))

function resize(h::Hamiltonian, θ::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    metric = h.metric
    if size(metric) != size(θ)
        metric = getname(metric)(size(θ))
        h = reconstruct(h, metric=metric)
    end
    return h
end

##
## Interface functions
##
function sample_init(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, 
    h::Hamiltonian, 
    θ::AbstractVecOrMat{<:AbstractFloat}
)
    # Ensure h.metric has the same dim as θ.
    h = resize(h, θ)
    # Initial transition
    t = Transition(phasepoint(rng, θ, h), NamedTuple())
    return h, t
end

function transition(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, 
    h::Hamiltonian, 
    κ::HMCKernel,
    z::PhasePoint,
)
    @unpack refreshment, τ = κ
    τ = reconstruct(τ, integrator=jitter(rng, τ.integrator))
    z = refresh(rng, refreshment, h, z)
    return transition(rng, τ, h, z)
end

Adaptation.adapt!(
    h::Hamiltonian,
    κ::AbstractMCMCKernel,
    adaptor::Adaptation.NoAdaptation,
    i::Int,
    n_adapts::Int,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
) = h, κ, false

function Adaptation.adapt!(
    h::Hamiltonian,
    κ::AbstractMCMCKernel,
    adaptor::AbstractAdaptor,
    i::Int,
    n_adapts::Int,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
)
    isadapted = false
    if i <= n_adapts
        i == 1 && Adaptation.initialize!(adaptor, n_adapts)
        adapt!(adaptor, θ, α)
        i == n_adapts && finalize!(adaptor)
        h = reconstruct(h, adaptor)
        κ = reconstruct(κ, adaptor)
        isadapted = true
    end
    return h, κ, isadapted
end

#################################
### AbstractMCMC.jl interface ###
#################################
struct HamiltonianModel{H} <: AbstractMCMC.AbstractModel
    hamiltonian :: H
end

struct HMCState{
    TTrans<:Transition,
    TAdapt<:Adaptation.AbstractAdaptor
}
    i::Int
    transition::TTrans
    adaptor::TAdapt
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::HamiltonianModel,
    spl::HMCKernel;
    init_params, # TODO: implement this. Just do `rand`? Need dimensionality though.
    adaptor::AbstractAdaptor=NoAdaptation(),
    kwargs...
)
    # Get an initial smaple
    h, t = AdvancedHMC.sample_init(rng, model.hamiltonian, init_params)

    # Compute next transition and state.
    state = HMCState(0, t, adaptor)

    # Take actual first step
    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::HamiltonianModel,
    spl::HMCKernel,
    state::HMCState;
    nadapts::Int=0,
    kwargs...
)
    # Get step size
    @debug "current ϵ" getstepsize(spl, state)

    # Compute transition.
    h = model.hamiltonian
    i = state.i + 1
    t_old = state.transition
    adaptor = state.adaptor

    # Make new transition.
    t = transition(rng, h, spl, t_old.z)

    # Adapt h and spl.
    tstat = stat(t)
    h, spl, isadapted = adapt!(h, spl, adaptor, i, nadapts, t.z.θ, tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt=isadapted,))

    # Compute next transition and state.
    newstate = HMCState(i, t, adaptor)

    # Return `Transition` with additional stats added.
    return Transition(t.z, tstat), newstate
end
