sample(h::Hamiltonian, prop::AbstractProposal, θ::AbstractVector{T}, n_samples::Int; verbose::Bool=true) where {T<:Real} =
    sample(GLOBAL_RNG, h, prop, θ, n_samples; verbose=verbose)

function sample(rng::AbstractRNG, h::Hamiltonian, prop::AbstractProposal, θ::AbstractVector{T}, n_samples::Int; verbose::Bool=true) where {T<:Real}
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    time = @elapsed for i = 1:n_samples
        θs[i], Hs[i], αs[i] = step(rng, h, prop, i == 1 ? θ : θs[i-1])
    end
    verbose && @info "Finished sampling with $time (s)" typeof(h.metric) typeof(prop) EBFMI(Hs) mean(αs)
    return θs
end

sample(h::Hamiltonian,
    prop::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int,
    adaptor::Adaptation.AbstractAdaptor,
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    verbose::Bool=true
) where {T<:Real} = sample(GLOBAL_RNG, h, prop, θ, n_samples, adaptor, n_adapts; verbose=verbose)

function sample(rng::AbstractRNG,
    h::Hamiltonian,
    prop::AbstractProposal,
    θ::AbstractVector{T},
    n_samples::Int,
    adaptor::Adaptation.AbstractAdaptor,
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    verbose::Bool=true
) where {T<:Real}
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    time = @elapsed for i = 1:n_samples
        θs[i], Hs[i], αs[i] = step(rng, h, prop, i == 1 ? θ : θs[i-1])
        if i <= n_adapts
            adapt!(adaptor, θs[i], αs[i])
            h, prop = update(h, prop, adaptor)
            if verbose
                if i == n_adapts
                    @info "Finished $n_adapts adapation steps" typeof(adaptor) prop.integrator.ϵ h.metric
                elseif i % Int(n_adapts / 10) == 0
                    @info "Adapting $i of $n_adapts steps" typeof(adaptor) prop.integrator.ϵ h.metric
                end
            end
        end
    end
    verbose && @info "Finished $n_samples sampling steps in $time (s)" typeof(h.metric) typeof(prop) EBFMI(Hs) mean(αs)
    return θs
end

function step(rng::AbstractRNG,
    h::Hamiltonian,
    prop::AbstractTrajectory{I},
    θ::AbstractVector{T}
) where {T<:Real,I<:AbstractIntegrator}
    h = update(h, θ) # Ensure h.metric has the same dim as θ.
    r = rand_momentum(rng, h)
    θ_new, r_new, α, H_new = transition(rng, prop, h, θ, r)
    return θ_new, H_new, α
end

step(h::Hamiltonian,
    p::AbstractTrajectory,
    θ::AbstractVector{T}
) where {T<:Real} = step(GLOBAL_RNG, h, p, θ)
