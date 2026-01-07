"""
    tr_product(A, B)

Compute `tr(A * B)` for square matrices in O(n²) without forming the product.
Uses the identity: tr(A * B) = sum(A' .* B)
"""
tr_product(A::AbstractMatrix, B::AbstractMatrix) = sum(Base.broadcasted(*, A', B))

"""
    tr_product(A, v)

Compute `tr(A * v * v')` = v' * A * v efficiently.
"""
tr_product(A::AbstractMatrix, v::AbstractVector) = dot(v, A * v)

####
#### Gradient cache for θ-dependent computations
####

"""
    RiemannianGradCache{T, TG, TP}

Cache for θ-dependent computations in Riemannian HMC gradient calculation.
This allows reusing expensive eigendecomposition/factorization across fixed-point iterations.

# Fields
- `G_eval`: Evaluated metric (SoftAbsEval or matrix)
- `∂P∂θ`: Pre-metric sensitivities, shape (d, d, d)
- `ℓπ`: Log density value at θ
- `∂ℓπ∂θ`: Log density gradient at θ
- `logdet_terms`: Precomputed 0.5 * tr(M_logdet * ∂P∂θ[:,:,i]) for each i
"""
struct RiemannianGradCache{T,TG,TP}
    G_eval::TG
    ∂P∂θ::TP
    ℓπ::T
    ∂ℓπ∂θ::Vector{T}
    logdet_terms::Vector{T}
end

"""
    build_grad_cache(h::Hamiltonian{<:AbstractRiemannianMetric}, θ)

Build cache for gradient computation at position θ.
Computes all θ-dependent quantities that can be reused across r values.
"""
function build_grad_cache(
    h::Hamiltonian{<:AbstractRiemannianMetric}, θ::AbstractVector{T}
) where {T}
    # Evaluate log density and gradient
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)

    # Evaluate metric and sensitivities
    G_eval = metric_eval(h.metric, θ)
    ∂P∂θ = metric_sensitivity(h.metric, θ)

    # Get logdet gradient matrix and precompute logdet gradient terms
    M_logdet = logdet_grad_matrix(G_eval)
    d = size(∂P∂θ, 3)
    logdet_terms = Vector{T}(undef, d)
    @inbounds for i in 1:d
        ∂Pᵢ = @view ∂P∂θ[:, :, i]
        logdet_terms[i] = T(0.5) * tr_product(M_logdet, ∂Pᵢ)
    end

    return RiemannianGradCache(G_eval, ∂P∂θ, ℓπ, ∂ℓπ∂θ, logdet_terms)
end

"""
    ∂H∂θ_from_cache(cache::RiemannianGradCache, r)

Compute Hamiltonian gradient ∂H/∂θ using cached θ-dependent values.
Only performs r-dependent computation (kinetic gradient matrix and trace products).
"""
function ∂H∂θ_from_cache(cache::RiemannianGradCache{T}, r::AbstractVector) where {T}
    # Compute kinetic gradient matrix (r-dependent)
    M_kinetic = kinetic_grad_matrix(cache.G_eval, r)

    # Compute full gradient
    d = length(cache.∂ℓπ∂θ)
    grad = Vector{T}(undef, d)

    @inbounds for i in 1:d
        ∂Pᵢ = @view cache.∂P∂θ[:, :, i]
        # ∂H/∂θᵢ = -∂ℓπ/∂θᵢ + 0.5*tr(M_logdet*∂P/∂θᵢ) - 0.5*tr(M_kinetic*∂P/∂θᵢ)
        kinetic_term = T(0.5) * tr_product(M_kinetic, ∂Pᵢ)
        grad[i] = -cache.∂ℓπ∂θ[i] + cache.logdet_terms[i] - kinetic_term
    end

    return DualValue(cache.ℓπ, grad)
end

####
#### Main gradient interface
####

"""
    ∂H∂θ(h::Hamiltonian{<:AbstractRiemannianMetric}, θ, r)

Compute the gradient of the Hamiltonian with respect to position θ.
Returns a DualValue containing (log_density, gradient).

Ref: Eq (15) of Girolami & Calderhead (2011)
"""
function ∂H∂θ(
    h::Hamiltonian{<:AbstractRiemannianMetric,<:GaussianKinetic},
    θ::AbstractVector,
    r::AbstractVector,
)
    cache = build_grad_cache(h, θ)
    return ∂H∂θ_from_cache(cache, r)
end

"""
    ∂H∂θ_cache(h, θ, r; cache=nothing)

Compute ∂H/∂θ with optional caching for fixed-point iterations.
Returns (DualValue, cache) tuple.

When cache is provided, reuses θ-dependent computations (eigendecomposition,
logdet gradient terms) and only recomputes r-dependent terms.
"""
function ∂H∂θ_cache(
    h::Hamiltonian{<:AbstractRiemannianMetric,<:GaussianKinetic},
    θ::AbstractVector,
    r::AbstractVector;
    cache=nothing,
)
    cache = @something cache build_grad_cache(h, θ)
    return ∂H∂θ_from_cache(cache, r), cache
end

####
#### Momentum gradient ∂H/∂r
####

"""
    ∂H∂r(h::Hamiltonian{<:AbstractRiemannianMetric}, θ, r; G_eval=nothing)

Compute the gradient of the Hamiltonian with respect to momentum r.
For Riemannian metrics: ∂H/∂r = G(θ)⁻¹ * r

If `G_eval` is provided, uses it directly instead of recomputing the metric.

Ref: Eq (14) of Girolami & Calderhead (2011)
"""
function ∂H∂r(
    h::Hamiltonian{<:AbstractRiemannianMetric,<:GaussianKinetic},
    θ::AbstractVector,
    r::AbstractVector;
    G_eval=nothing,
)
    G = @something G_eval metric_eval(h.metric, θ)
    return G \ r
end

# Non-keyword version for backward compatibility with integrator
function ∂H∂r(
    h::Hamiltonian{<:AbstractRiemannianMetric,<:GaussianKinetic},
    θ::AbstractVector,
    r::AbstractVector,
)
    G_eval = metric_eval(h.metric, θ)
    return G_eval \ r
end

####
#### Negative energy (log probability)
####

"""
    neg_energy(h::Hamiltonian{<:AbstractRiemannianMetric}, r, θ; G_eval=nothing)

Compute the negative kinetic energy for Riemannian metrics.
Includes the log-determinant normalization term since G depends on θ.

If `G_eval` is provided, uses it directly instead of recomputing the metric.

K(r, θ) = 0.5 * (D*log(2π) + log|G(θ)| + r'G(θ)⁻¹r)
neg_energy = -K = -0.5 * (D*log(2π) + log|G(θ)| + r'G(θ)⁻¹r)

Ref: Eq (13) of Girolami & Calderhead (2011)
"""
function neg_energy(
    h::Hamiltonian{<:AbstractRiemannianMetric,<:GaussianKinetic},
    r::AbstractVector,
    θ::AbstractVector;
    G_eval=nothing,
)
    G = @something G_eval metric_eval(h.metric, θ)
    D = length(r)

    # Quadratic form: r' * G⁻¹ * r
    G_inv_r = G \ r
    quadform = dot(r, G_inv_r)

    # Log normalization constant (position-dependent)
    logZ = (D * log(2π) + logdet(G)) / 2

    return -logZ - quadform / 2
end

# Non-keyword version for backward compatibility
function neg_energy(
    h::Hamiltonian{<:AbstractRiemannianMetric,<:GaussianKinetic},
    r::AbstractVector,
    θ::AbstractVector,
)
    G_eval = metric_eval(h.metric, θ)
    return neg_energy(h, r, θ; G_eval=G_eval)
end

####
#### Phase point construction
####

"""
Create a PhasePoint for Riemannian metrics, computing position-dependent kinetic energy.
Shares the metric evaluation between neg_energy and ∂H∂r to avoid redundant computation.
"""
function phasepoint(
    h::Hamiltonian{<:AbstractRiemannianMetric},
    θ::T,
    r::T;
    ℓπ=∂H∂θ(h, θ, r),
    G_eval=nothing,
    ℓκ=nothing,
) where {T<:AbstractVecOrMat}
    if isnothing(ℓκ)
        # Compute G_eval once and share between neg_energy and ∂H∂r
        G = @something G_eval metric_eval(h.metric, θ)
        ℓκ = DualValue(neg_energy(h, r, θ; G_eval=G), ∂H∂r(h, θ, r; G_eval=G))
    end
    return PhasePoint(θ, r, ℓπ, ℓκ)
end
