function _is_accept(H::Real, H_new::Real)
    return log(rand()) + H_new < min(H_new, H), min(0, -(H_new - H))
end

function sample(h::Hamiltonian, t::AbstractTrajectory, θ::AbstractVector{T}, n_samples::Integer) where {T<:Real}
    samples = Vector{Vector{T}}(undef, n_samples)
    for n = 1:n_samples
        θ = step(h, t, θ)
        samples[n] = θ
    end
    return samples
end

# HMC
function step(h::Hamiltonian, st::StaticTrajectory, θ::AbstractVector{T}) where {T<:Real}
    r = rand_momentum(h, θ)
    H = _H(h, θ, r)
    θ_new, r_new = build_and_sample(st, h, θ, r)
    H_new = _H(h, θ_new, r_new)
    is_accept, _ = _is_accept(H, H_new)
    if is_accept
        θ = θ_new
    end
    return θ
end

# # Constant used in the base case of `build_tree`
# # 1000 is the recommended value from Hoffman et al. (2011)
# const Δ_max = 1000
