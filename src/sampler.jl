function _logα(H::Real, H_new::Real)
    return min(0, H - H_new)
end

function is_accept(logα::Real)
    return log(rand()) < logα
end

function sample(h::Hamiltonian, p::AbstractProposal, θ::AbstractVector{T}, n_samples::Integer) where {T<:Real}
    θs = Vector{Vector{T}}(undef, n_samples)
    Hs = Vector{T}(undef, n_samples)
    αs = Vector{T}(undef, n_samples)
    time = @elapsed for i = 1:n_samples
        θs[i], Hs[i], αs[i] = step(h, p, i == 1 ? θ : θs[i-1])
    end
    @info "Finished sampling with $time (s)" typeof(h) typeof(p) EBFMI(Hs) mean(αs)
    return θs
end

# HMC is just one speical example with static trajectory
function step(h::Hamiltonian, p::AbstractProposal{StaticTrajectory{I}}, θ::AbstractVector{T}) where {T<:Real,I<:AbstractIntegrator}
    r = rand_momentum(h, θ)
    H = _H(h, θ, r)
    θ_new, r_new = propose(p, h, θ, r)
    H_new = _H(h, θ_new, r_new)
    logα = _logα(H, H_new)
    α = exp(logα)
    if !is_accept(logα)
        return θ, H, α
    end
    return θ_new, H_new, α
end

# # Constant used in the base case of `build_tree`
# # 1000 is the recommended value from Hoffman et al. (2011)
# const Δ_max = 1000
