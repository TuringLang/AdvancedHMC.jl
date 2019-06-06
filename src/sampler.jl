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
    # Prepare phase point for sampling
    h = update(h, θ) # Ensure h.metric has the same dim as θ.
    r = rand(rng, h.metric)
    z = phasepoint(h, θ, r)
    pm = progress ? Progress(n_samples, desc="Sampling", barlen=31) : nothing
    time = @elapsed for i = 1:n_samples
        z, αs[i] = transition(rng, τ, h, z)
        θs[i], Hs[i] = z.θ, neg_energy(z)
        progress && (showvalues = Tuple[(:iteration, i), (:hamiltonian_energy, Hs[i]), (:acceptance_rate, αs[i])])
        if !(adaptor isa Adaptation.NoAdaptation)
            if i <= n_adapts
                adapt!(adaptor, θs[i], αs[i])
                # Finalize adapation
                if i == n_adapts
                    finalize!(adaptor)
                    verbose && @info "Finished $n_adapts adapation steps" typeof(adaptor) τ.integrator.ϵ h.metric
                end
                h, τ = update(h, τ, adaptor)
            end
            # Progress info for adapation
            progress && append!(showvalues, [(:step_size, τ.integrator.ϵ), (:precondition, h.metric)])
        end
        progress && ProgressMeter.next!(pm; showvalues=showvalues)
        # Refresh momentum for next iteration
        z = rand_momentum(rng, z, h)
    end
    verbose && @info "Finished $n_samples sampling steps in $time (s)" typeof(h.metric) typeof(τ) EBFMI(Hs) mean(αs)
    return θs
end
