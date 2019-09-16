##
## Interface functions
##

function sample_init(rng::AbstractRNG, h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    # Ensure h.metric has the same dim as θ.
    h = update(h, θ)
    # Initial transition
    t = Transition(phasepoint(rng, θ, h), NamedTuple())
    return h, t
end

# A step is a momentum refreshment plus a transition
function step(rng::AbstractRNG, h::Hamiltonian, τ::AbstractProposal, z::PhasePoint)
    # Refresh momentum
    z = refresh(rng, z, h)
    # Make transition
    return transition(rng, τ, h, z)
end

adapt!(
    h::Hamiltonian,
    τ::AbstractProposal,
    adaptor::Adaptation.NoAdaptation,
    i::Int,
    n_adapts::Int,
    θ::AbstractVector{T},
    α::T
) where {T<:Real} = h, τ, false

function adapt!(
    h::Hamiltonian,
    τ::AbstractProposal,
    adaptor::Adaptation.AbstractAdaptor,
    i::Int,
    n_adapts::Int,
    θ::AbstractVector{T},
    α::T
) where {T<:Real}
    isadapted = false
    if i <= n_adapts
        adapt!(adaptor, θ, α)
        i == n_adapts && finalize!(adaptor)
        h, τ = update(h, τ, adaptor)
        isadapted = true
    end
    return h, τ, isadapted
end

"""
Progress meter update with all trajectory stats, iteration number and metric shown.
"""
function pm_next!(pm, stat::NamedTuple, i::Int, metric::AbstractMetric)
    # Add current iteration and mass matrix
    stat = (iterations=i, stat..., mass_matrix=metric)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

"""
Simple progress meter update without any show values.
"""
simple_pm_next!(pm, stat::NamedTuple, ::Int, ::AbstractMetric) = ProgressMeter.next!(pm)

##
## Sampling functions
##

sample(
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int,
    adaptor::Adaptation.AbstractAdaptor=Adaptation.NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) where {T<:Real} = sample(GLOBAL_RNG, h, τ, θ, n_samples, adaptor, n_adapts; verbose=verbose, progress=progress, (pm_next!)=pm_next!)

"""
    sample(
        rng::AbstractRNG,
        h::Hamiltonian,
        τ::AbstractProposal,
        θ::AbstractVector{T},
        n_samples::Int,
        adaptor::Adaptation.AbstractAdaptor=Adaptation.NoAdaptation(),
        n_adapts::Int=min(div(n_samples, 10), 1_000);
        verbose::Bool=true,
        progress::Bool=false
    )

Sample `n_samples` samples using the proposal `τ` under Hamiltonian `h`.
- the initial point is given by `θ`
- the randomness is controlled by `rng`
- the adaptor is set by `adaptor`, for which the default is no adapation
    - it will perform `n_adapts` steps of adapations, for which the default is the minimum of `1_000` and 10% of `n_samples`
- the verbosity is controlled by the boolean variable `verbose` and
- the visibility of the progress meter is controlled by the bollean variable `progress`
"""
function sample(
    rng::AbstractRNG,
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int,
    adaptor::Adaptation.AbstractAdaptor=Adaptation.NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) where {T<:Real}
    # Prepare containers to store sampling results
    θs, stats = Vector{Vector{T}}(undef, n_samples), Vector{NamedTuple}(undef, n_samples)
    # Initial sampling
    h, t = sample_init(rng, h, θ)
    # Progress meter
    pm = progress ? Progress(n_samples, desc="Sampling", barlen=31) : nothing
    time = @elapsed for i = 1:n_samples
        # Make a step
        t = step(rng, h, τ, t.z)
        # Adapt h and τ; what mutable is the adaptor
        h, τ, isadapted = adapt!(h, τ, adaptor, i, n_adapts, t.z.θ, t.stat.acceptance_rate)
        # Update progress meter
        if progress
            pm_next!(pm, t.stat, i, h.metric)
        # Report finish of adapation
        elseif verbose && isadapted && i == n_adapts
            @info "Finished $n_adapts adapation steps" adaptor τ.integrator h.metric
        end
        # Store sample
        θs[i], stats[i] = t.z.θ, t.stat
    end
    # Report end of sampling
    if verbose
        EBFMI_est = EBFMI(map(s -> s.hamiltonian_energy, stats)) 
        average_acceptance_rate = mean(map(s -> s.acceptance_rate, stats))
        @info "Finished $n_samples sampling steps in $time (s)" h τ EBFMI_est average_acceptance_rate
    end
    return θs, stats
end
