#! Eq (14) of Girolami & Calderhead (2011)
function ∂H∂r(
    h::Hamiltonian{<:DenseRiemannianMetric,<:GaussianKinetic}, θ::AbstractVecOrMat, r::AbstractVecOrMat
)
    H = h.metric.G(θ)
    G = h.metric.map(H)
    return G \ r # NOTE it's actually pretty weird that ∂H∂θ returns DualValue but ∂H∂r doesn't
end

function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:IdentityMap},<:GaussianKinetic},
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
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap},<:GaussianKinetic},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    return ∂H∂θ_cache(h, θ, r)
end
function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap},<:GaussianKinetic},
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

        R = diagm(1 ./ softabsλ)

        # softabsΛ = diagm(softabsλ)
        # M = inv(softabsΛ) * Q' * r
        # M = R * Q' * r # equiv to above but avoid inv

        J = make_J(λ, h.metric.map.α)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        term_1_cached = Q * (R .* J) * Q'
    else
        ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached = cache
    end
    d = length(∂ℓπ∂θ)
    D = diagm((Q' * r) ./ softabsλ)
    term_2_cached = Q * D * J * D * Q'
    g =
        isdiag ?
        -(∂ℓπ∂θ - 1 / 2 * diag(term_1_cached * ∂H∂θ) + 1 / 2 * diag(term_2 * ∂H∂θ)) :
        -mapreduce(vcat, 1:d) do i
            ∂H∂θᵢ = ∂H∂θ[:, :, i]
            # ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * M' * (J .* (Q' * ∂H∂θᵢ * Q)) * M # (v1)
            # NOTE Some further optimization can be done here: cache the 1st product all together
            ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * tr(term_2_cached * ∂H∂θᵢ) # (v2) cache friendly
        end

    dv = DualValue(ℓπ, g)
    return return_cache ? (dv, (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached)) : dv
end

# QUES Do we want to change everything to position dependent by default?
# Add θ to ∂H∂r for DenseRiemannianMetric
function phasepoint(
    h::Hamiltonian{<:DenseRiemannianMetric},
    θ::T,
    r::T;
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, θ, r)),
) where {T<:AbstractVecOrMat}
    return PhasePoint(θ, r, ℓπ, ℓκ)
end

#! Eq (13) of Girolami & Calderhead (2011)
function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric,<:GaussianKinetic}, r::T, θ::T
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    D = size(G, 1)
    # Need to consider the normalizing term as it is no longer same for different θs
    logZ = 1 / 2 * (D * log(2π) + logdet(G)) # it will be user's responsibility to make sure G is SPD and logdet(G) is defined
    mul!(h.metric._temp, inv(G), r)
    return -logZ - dot(r, h.metric._temp) / 2
end
