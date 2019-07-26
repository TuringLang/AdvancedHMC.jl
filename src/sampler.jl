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
    # Prepare phase point for sampling
    h = update(h, θ) # Ensure h.metric has the same dim as θ.
    r = rand(rng, h.metric)
    z = phasepoint(h, θ, r)
    pm = progress ? Progress(n_samples, desc="Sampling", barlen=31) : nothing
    time = @elapsed for i = 1:n_samples
        z, stat = transition(rng, τ, h, z)
        θs[i], αs[i], Hs[i], stats[i] = z.θ, stat.acceptance_rate, stat.hamiltonian_energy, stat
        showvalues = Dict{Symbol,Any}(pairs(stat)...)
        if !(adaptor isa Adaptation.NoAdaptation)
            if i <= n_adapts
                adapt!(adaptor, θs[i], αs[i])
                i == n_adapts && finalize!(adaptor)
                h, τ = update(h, τ, adaptor)
                (i == n_adapts && verbose && !progress) && @info "Finished $n_adapts adapation steps" adaptor τ.integrator h.metric
            end
            # Progress info for adapation
            progress && (showvalues[:step_size] = τ.integrator.ϵ; showvalues[:precondition] = h.metric)
        end
        # Refresh momentum for next iteration
        z = refresh(rng, z, h)
        progress && ProgressMeter.next!(pm; showvalues=Tuple[(:iteration, i), zip(keys(showvalues), values(showvalues))...])
    end
    # Report end of sampling
    verbose && @info "Finished $n_samples sampling steps in $time (s)" h τ EBFMI(Hs) mean(αs)
    return θs, stats
end
