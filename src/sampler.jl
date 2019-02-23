function _logα(H::Real, H_new::Real)
    return min(0, H - H_new)
end

function is_accept(logα::Real)
    return log(rand()) < logα
end

function sample(h::Hamiltonian, p::AbstractProposal, θ::AbstractVector{T}, n_samples::Integer) where {T<:Real}
    samples = Vector{Vector{T}}(undef, n_samples)
    Es = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    for n = 1:n_samples
        θ, H , α = step(h, p, θ)
        samples[n] = θ
        Es[n] = H
        αs[n] = α
    end
    @info "Sampling statistics" EBFMI(Es) mean(αs)
    return samples
end

# HMC
function step(h::Hamiltonian, p::AbstractProposal, θ::AbstractVector{T}) where {T<:Real}
    r = rand_momentum(h, θ)
    H = _H(h, θ, r)
    θ_new, r_new = propose(p, h, θ, r)
    H_new = _H(h, θ_new, r_new)
    logα = _logα(H, H_new)
    if is_accept(logα)
        θ = θ_new
        H = H_new
    end
    return θ, H, exp(logα)
end

# # Constant used in the base case of `build_tree`
# # 1000 is the recommended value from Hoffman et al. (2011)
# const Δ_max = 1000
