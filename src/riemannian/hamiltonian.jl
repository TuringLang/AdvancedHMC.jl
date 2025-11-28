import AdvancedHMC: refresh, phasepoint, neg_energy, ∂H∂θ, ∂H∂r
using AdvancedHMC: FullMomentumRefreshment, PartialMomentumRefreshment, DualValue, PhasePoint
using LinearAlgebra: logabsdet, tr, diagm, logdet

# Specialized phasepoint for Riemannian metrics that need θ for momentum gradient
function phasepoint(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    θ::AbstractVecOrMat{T},
    h::Hamiltonian,
) where {T<:Real}
    return phasepoint(h, θ, rand_momentum(rng, h.metric, h.kinetic, θ))
end

# To change L191 of hamiltonian.jl
function refresh(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    ::FullMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
)
    return phasepoint(h, z.θ, rand_momentum(rng, h.metric, h.kinetic, z.θ))
end

# To change L215 of hamiltonian.jl
function refresh(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    ref::PartialMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
)
    return phasepoint(
        h,
        z.θ,
        ref.α * z.r + sqrt(1 - ref.α^2) * rand_momentum(rng, h.metric, h.kinetic, z.θ),
    )
end

###
### DenseRiemannianMetric-specific Hamiltonian methods
###

# Specialized phasepoint for DenseRiemannianMetric that passes θ to ∂H∂r
function phasepoint(
    h::Hamiltonian{<:DenseRiemannianMetric},
    θ::T,
    r::T;
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, θ, r)),
) where {T<:AbstractVecOrMat}
    return PhasePoint(θ, r, ℓπ, ℓκ)
end

# Negative kinetic energy
#! Eq (13) of Girolami & Calderhead (2011)
function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric}, r::T, θ::T
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    D = size(G, 1)
    # Need to consider the normalizing term as it is no longer same for different θs
    logZ = 1 / 2 * (D * log(2π) + logdet(G)) # it will be user's responsibility to make sure G is SPD and logdet(G) is defined
    mul!(h.metric._temp, inv(G), r)
    return -logZ - dot(r, h.metric._temp) / 2
end

# Position gradient with Riemannian correction terms
function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:IdentityMap}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
    G = h.metric.map(h.metric.G(θ))
    invG = inv(G)
    ∂G∂θ = h.metric.∂G∂θ(θ)
    d = length(∂ℓπ∂θ)
    return DualValue(
        ℓπ,
        #! Eq (15) of Girolami & Calderhead (2011)
        -mapreduce(vcat, 1:d) do i
            ∂G∂θᵢ = ∂G∂θ[:, :, i]
            ∂ℓπ∂θ[i] - 1 / 2 * tr(invG * ∂G∂θᵢ) + 1 / 2 * r' * invG * ∂G∂θᵢ * invG * r
            # Gr = G \ r
            # ∂ℓπ∂θ[i] - 1 / 2 * tr(G \ ∂G∂θᵢ) + 1 / 2 * Gr' * ∂G∂θᵢ * Gr
            # 1 / 2 * tr(invG * ∂G∂θᵢ)
            # 1 / 2 * r' * invG * ∂G∂θᵢ * invG * r
        end,
    )
end

# Ref: https://www.wolframalpha.com/input?i=derivative+of+x+*+coth%28a+*+x%29
#! Based on middle of the right column of Page 3 of Betancourt (2012) "Note that whenλi=λj, such as for the diagonal elementsor degenerate eigenvalues, this becomes the derivative"
dsoftabsdλ(α, λ) = coth(α * λ) + λ * α * -csch(λ * α)^2

#! J as defined in middle of the right column of Page 3 of Betancourt (2012)
function make_J(λ::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    d = length(λ)
    J = Matrix{T}(undef, d, d)
    for i in 1:d, j in 1:d
        J[i, j] = if (λ[i] == λ[j])
            dsoftabsdλ(α, λ[i])
        else
            ((λ[i] * coth(α * λ[i]) - λ[j] * coth(α * λ[j])) / (λ[i] - λ[j]))
        end
    end
    return J
end

function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    return ∂H∂θ_cache(h, θ, r)
end

function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache=false,
    cache=nothing,
) where {T}
    # Terms that only dependent on θ can be cached in θ-unchanged loops
    if isnothing(cache)
        ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
        H = h.metric.G(θ)
        ∂H∂θ = h.metric.∂G∂θ(θ)

        G, Q, λ, softabsλ = softabs(H, h.metric.map.α)

        R = Diagonal(1 ./ softabsλ)

        # softabsΛ = diagm(softabsλ)
        # M = inv(softabsΛ) * Q' * r
        # M = R * Q' * r # equiv to above but avoid inv

        J = make_J(λ, h.metric.map.α)

        tmp1 = similar(H)
        tmp2 = similar(H)
        tmp3 = similar(H)
        tmp4 = similar(softabsλ)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        tmp1 = R .* J
        # tmp2 = Q * tmp1
        mul!(tmp2, Q, tmp1)

        # tmp1 = tmp2 * Q'
        mul!(tmp1, tmp2, Q')

        term_1_cached = tmp1

        # Cache first part of the equation
        term_1_prod = similar(∂ℓπ∂θ)
        @inbounds for i in 1:length(∂ℓπ∂θ)
            ∂H∂θᵢ = ∂H∂θ[:, :, i]
            term_1_prod[i] = ∂ℓπ∂θ[i] - 1/2 * tr(term_1_cached * ∂H∂θᵢ)
        end

    else
        ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_prod, tmp1, tmp2, tmp3, tmp4 = cache
    end
    d = length(∂ℓπ∂θ)
    mul!(tmp4, Q', r)
    D = Diagonal(tmp4 ./ softabsλ)

    # tmp1 = D * J
    mul!(tmp1, D, J)
    # tmp2 = tmp1 * D
    mul!(tmp2, tmp1, D)
    # tmp1 = Q * tmp2
    mul!(tmp1, Q, tmp2)
    # tmp2 = tmp1 * Q'
    mul!(tmp2, tmp1, Q')
    term_2_cached = tmp2

    # g =
    #     -mapreduce(vcat, 1:d) do i
    #         ∂H∂θᵢ = ∂H∂θ[:, :, i]
    #         # ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * M' * (J .* (Q' * ∂H∂θᵢ * Q)) * M # (v1)
    #         # NOTE Some further optimization can be done here: cache the 1st product all together
    #         ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * tr(term_2_cached * ∂H∂θᵢ) # (v2) cache friendly
    #     end
    g = similar(∂ℓπ∂θ)
    @inbounds for i in 1:d
        ∂H∂θᵢ = ∂H∂θ[:, :, i]
        g[i] = term_1_prod[i] + 1/2 * tr(term_2_cached * ∂H∂θᵢ)
    end
    g .*= -1

    dv = DualValue(ℓπ, g)
    return return_cache ? (dv, (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_prod, tmp1, tmp2, tmp3, tmp4)) : dv
end

#! Eq (14) of Girolami & Calderhead (2011)
function ∂H∂r(
    h::Hamiltonian{<:DenseRiemannianMetric}, θ::AbstractVecOrMat{T}, r::AbstractVecOrMat{T}
) where {T}
    H = h.metric.G(θ)
    # if !all(isfinite, H)
    #     println("θ: ", θ)
    #     println("H: ", H)
    # end
    G = h.metric.map(H)
    # return inv(G) * r
    # println("G \ r: ", G \ r)
    return G \ r # NOTE it's actually pretty weird that ∂H∂θ returns DualValue but ∂H∂r doesn't
end
