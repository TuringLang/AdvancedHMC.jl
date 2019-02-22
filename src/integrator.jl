# NOTE: we may not want this type
abstract type AbstractIntegrator end

struct Leapfrog{T<:Real} <: AbstractIntegrator
    ϵ   ::  T
end

function step(l::Leapfrog{T}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    r = r - l.ϵ / 2 .* _dHdθ(h, θ)
    θ = θ + l.ϵ     .* _dHdr(h, r)
    r = r - l.ϵ / 2 .* _dHdθ(h, θ)
    return θ, r
end

# TODO: improve below
function step(l::Leapfrog{T}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}, n_steps::Integer) where {T<:Real}
    for _ = 1:n_steps
        θ, r = step(l, h, θ, r)
    end
    return θ, r
end
