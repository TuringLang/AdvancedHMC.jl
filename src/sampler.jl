##
## Sampling functions
##

sample(
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int;
    verbose::Bool=true
) where {T<:Real} = sample(GLOBAL_RNG, h, τ, θ, n_samples; verbose=verbose)

function sample(
    rng::AbstractRNG,
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int;
    verbose::Bool=true
) where {T<:Real}
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    r = rand(rng, h.metric)
    z = phasepoint(h, θ, r)
    time = @elapsed for i = 1:n_samples
        # θs[i], _, αs[i], Hs[i] = transition(rng, τ, h, i == 1 ? θ : θs[i-1], r)
        z, αs[i] = transition(rng, τ, h, z)
        θs[i], Hs[i] = z.θ, neg_energy(z)
        z = rand_momentum(rng, z, h)
    end
    verbose && @info "Finished sampling with $time (s)" typeof(h.metric) typeof(τ) EBFMI(Hs) mean(αs)
    return θs
end

sample(
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int,
    adaptor::Adaptation.AbstractAdaptor,
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    verbose::Bool=true
) where {T<:Real} = sample(GLOBAL_RNG, h, τ, θ, n_samples, adaptor, n_adapts; verbose=verbose)

function sample(
    rng::AbstractRNG,
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int,
    adaptor::Adaptation.AbstractAdaptor,
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    verbose::Bool=true
) where {T<:Real}
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    r = rand(rng, h.metric)
    z = phasepoint(h, θ, r)
    time = @elapsed for i = 1:n_samples
        # θs[i], Hs[i], αs[i] = step(rng, h, τ, i == 1 ? θ : θs[i-1])
        # θs[i], _, αs[i], Hs[i] = transition(rng, τ, h, i == 1 ? θ : θs[i-1], r)
        z, αs[i] = transition(rng, τ, h, z)
        θs[i], Hs[i] = z.θ, neg_energy(z)
        if i <= n_adapts
            adapt!(adaptor, θs[i], αs[i])
            h, τ = update(h, τ, adaptor)
            if verbose
                if i == n_adapts
                    @info "Finished $n_adapts adapation steps" typeof(adaptor) τ.integrator.ϵ h.metric
                elseif i % Int(n_adapts / 10) == 0
                    @info "Adapting $i of $n_adapts steps" typeof(adaptor) τ.integrator.ϵ h.metric
                end
            end
        end
        z = rand_momentum(rng, z, h)
    end
    verbose && @info "Finished $n_samples sampling steps in $time (s)" typeof(h.metric) typeof(τ) EBFMI(Hs) mean(αs)
    return θs
end
