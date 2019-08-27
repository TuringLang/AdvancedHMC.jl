##
## Interface functions
##

function init(rng::AbstractRNG, h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    # Ensure h.metric has the same dim as θ.
    h = update(h, θ)
    # Prepare phase point for sampling
    r = rand(rng, h.metric)
    z = phasepoint(h, θ, r)
    return h, z
end

function step(rng::AbstractRNG, h::Hamiltonian, τ::AbstractProposal, z::PhasePoint)
    # Make transition
    z, stat = transition(rng, τ, h, z)
    # Collect stats
    θ, α, H = z.θ, stat.acceptance_rate, stat.hamiltonian_energy
    # Refresh momentum for next iteration
    z = refresh(rng, z, h)
    return z, θ, α, H, stat
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

function ProgressMeter.next!(pm, stat::NamedTuple, i::Int, metric)
    # Add current iteration and mass matrix
    stat = (iterations=i, stat..., mass_matrix=metric)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

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
    progress::Bool=false
) where {T<:Real} = sample(GLOBAL_RNG, h, τ, θ, n_samples, adaptor, n_adapts; verbose=verbose, progress=progress)

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
    progress::Bool=false
) where {T<:Real}
    # Prepare containers to store sampling results
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    stats = Vector{NamedTuple}(undef, n_samples)
    h, z = init(rng, h, θ)
    # Progress meter
    pm = progress ? Progress(n_samples, desc="Sampling", barlen=31) : nothing
    time = @elapsed for i = 1:n_samples
        z, θs[i], αs[i], Hs[i], stats[i] = step(rng, h, τ, z)
        # Adapt h and τ; what mutable is the adaptor
        h, τ, isadapted = adapt!(h, τ, adaptor, i, n_adapts, θs[i], αs[i])
        # Adapation finish
        if isadapted && i == n_adapts && verbose && !progress
            @info "Finished $n_adapts adapation steps" adaptor τ.integrator h.metric
        end
        progress && ProgressMeter.next!(pm, stats[i], i, h.metric)
    end
    # Report end of sampling
    verbose && @info "Finished $n_samples sampling steps in $time (s)" h τ EBFMI(Hs) mean(αs)
    return θs, stats
end
