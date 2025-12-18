#! Eq (14) of Girolami & Calderhead (2011)
"The gradient of the Hamiltonian with respect to the momentum."
function ∂H∂r(
    h::Hamiltonian{<:DenseRiemannianMetric,<:GaussianKinetic},
    θ::AbstractVector,
    r::AbstractVector,
)
    H = h.metric.G(θ)
    G = h.metric.map(H)
    return G \ r
end

"""
Computes `tr(A*B)` for square n x n matrices `A` and `B` in O(n^2) without computing `A*B`, which would be O(n^3).

Doesn't actually check that A and B are both n x n matrices.
"""
tr_product(A::AbstractMatrix, B::AbstractMatrix) = sum(Base.broadcasted(*, A', B))
"Computes `tr(A*v*v')`, i.e. dot(v,A,v)."
tr_product(A::AbstractMatrix, v::AbstractVector) = sum(Base.broadcasted(*, v, A, v'))


function ∂H∂θ(
    h::Hamiltonian{<:AbstractRiemannianMetric,<:GaussianKinetic},
    θ::AbstractVector,
    r::AbstractVector,
)
    return first(∂H∂θ_cache(h, θ, r))
end
"""

"""
@views function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:IdentityMap},<:GaussianKinetic},
    θ::AbstractVector{T},
    r::AbstractVector{T};
    cache=nothing
) where {T}
    cache = @something cache begin 
        log_density, log_density_gradient = h.∂ℓπ∂θ(θ)
        # h.metric.map is the IdentityMap
        metric = h.metric.G(θ)
        # The metric is inverted to be able to compute `tr_product(inv_metric, ...)` efficiently -
        # but this may still be a bad idea!
        inv_metric = inv(metric)
        metric_sensitivities = h.metric.∂G∂θ(θ)
        rv1 = map(eachindex(log_density_gradient)) do i 
            -log_density_gradient[i] + .5 * tr_product(inv_metric, metric_sensitivities[:, :, i])
        end
        (;log_density, inv_metric, metric_sensitivities, rv1)
    end
    # (;log_density, inv_metric_r, metric_sensitivities, rv1) = cache
    inv_metric_r = cache.inv_metric * r
    return DualValue(
        cache.log_density,
        #! Eq (15) of Girolami & Calderhead (2011)
        cache.rv1 .- Base.broadcasted(eachindex(cache.rv1)) do i 
            .5 * tr_product(cache.metric_sensitivities[:, :, i], inv_metric_r)
        end
    ), cache
end

#! J as defined in middle of the right column of Page 3 of Betancourt (2012)
function make_J(λ::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    d = length(λ)
    J = Matrix{T}(undef, d, d)
    for i in 1:d, j in 1:d
        J[i, j] = if (λ[i] == λ[j])
            # Ref: https://www.wolframalpha.com/input?i=derivative+of+x+*+coth%28a+*+x%29
            #! Based on middle of the right column of Page 3 of Betancourt (2012) "Note that whenλi=λj, such as for the diagonal elementsor degenerate eigenvalues, this becomes the derivative"
            coth(α * λ[i]) + λ[i] * α * -csch(λ[i] * α)^2
        else
            ((λ[i] * coth(α * λ[i]) - λ[j] * coth(α * λ[j])) / (λ[i] - λ[j]))
        end
    end
    return J
end

@views function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap},<:GaussianKinetic},
    θ::AbstractVector{T},
    r::AbstractVector{T};
    cache=nothing,
) where {T}
    cache = @something cache begin 
        log_density, log_density_gradient = h.∂ℓπ∂θ(θ)
        premetric = h.metric.G(θ)
        premetric_sensitivities = h.metric.∂G∂θ(θ)
        metric, Q, λ, softabsλ = softabs(premetric, h.metric.map.α)
        J = make_J(λ, h.metric.map.α)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        tmpv = diag(J) ./ softabsλ
        tmpm = Q * Diagonal(tmpv) * Q'

        rv1 = map(eachindex(log_density_gradient)) do i 
            -log_density_gradient[i] + .5 * tr_product(tmpm, premetric_sensitivities[:, :, i])
        end
        (;log_density, Q, softabsλ, tmpv, tmpm, rv1)
    end
    cache.tmpv .= (cache.Q' * r) ./ cache.softabsλ
    cache.tmpm .= Q * (J .* cache.tmpv .* cache.tmpv') * Q'

    return DualValue(
        cache.log_density,
        cache.rv1 .- Base.broadcasted(eachindex(cache.rv1)) do i 
            .5 * tr_product(cache.tmpm, cache.premetric_sensitivities[:, :, i])
        end
    ), cache
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
