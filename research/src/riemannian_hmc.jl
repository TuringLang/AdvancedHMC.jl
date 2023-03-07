using Random

### integrator.jl

import AdvancedHMC: ∂H∂θ, ∂H∂r, DualValue, PhasePoint, phasepoint, step
using AdvancedHMC: @unpack, TYPEDEF, TYPEDFIELDS, AbstractScalarOrVec, AbstractLeapfrog, step_size

"""
$(TYPEDEF)

Generalized leapfrog integrator with fixed step size `ϵ`.

# Fields

$(TYPEDFIELDS)
"""
struct GeneralizedLeapfrog{T<:AbstractScalarOrVec{<:AbstractFloat}} <: AbstractLeapfrog{T}
    "Step size."
    ϵ::T
    n::Int
end
Base.show(io::IO, l::GeneralizedLeapfrog) = print(io, "GeneralizedLeapfrog(ϵ=$(round.(l.ϵ; sigdigits=3)), n=$(l.n))")

# TODO Make sure vectorization works
# TODO Check if tempering is valid
function step(
    lf::GeneralizedLeapfrog{T},
    h::Hamiltonian,
    z::P,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj} = Val(false)
) where {T<:AbstractScalarOrVec{<:AbstractFloat}, P<:PhasePoint, FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? step_size(lf) : -step_size(lf)
    ϵ = ϵ'

    res = if FullTraj
        Vector{P}(undef, n_steps)
    else
        z
    end

    for i = 1:n_steps
        θ_init, r_init = z.θ, z.r
        # Tempering
        #r = temper(lf, r, (i=i, is_half=true), n_steps)
        #! Eq (16) of Girolami & Calderhead (2011)
        r_half = copy(r_init)
        for j in 1:lf.n
            # Reuse cache for the first iteration
            if j == 1
                @unpack value, gradient = z.ℓπ
            else
                @unpack value, gradient = ∂H∂θ(h, θ_init, r_half)
            end
            r_half = r_init - ϵ / 2 * gradient
            # println("r_half: ", r_half)
        end
        #! Eq (17) of Girolami & Calderhead (2011)
        θ_full = copy(θ_init)
        for j in 1:lf.n
            θ_full = θ_init + ϵ / 2 * (∂H∂r(h, θ_init, r_half) + ∂H∂r(h, θ_full, r_half))
            # println("θ_full :", θ_full)
        end
        #! Eq (18) of Girolami & Calderhead (2011)
        @unpack value, gradient = ∂H∂θ(h, θ_full, r_half)
        r_full = r_half - ϵ / 2 * gradient
        # println("r_full: ", r_full)
        # Tempering
        #r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ_full, r_full; ℓπ=DualValue(value, gradient))
        # Update result
        if FullTraj
            res[i] = z
        else
            res = z
        end
        if !isfinite(z)
            # Remove undef
            if FullTraj
                res = res[isassigned.(Ref(res), 1:n_steps)]
            end
            break
        end
        # @assert false
    end
    return res
end

# TODO Make the order of θ and r consistent with neg_energy
∂H∂θ(h::Hamiltonian, θ::AbstractVecOrMat, r::AbstractVecOrMat) = ∂H∂θ(h, θ)
∂H∂r(h::Hamiltonian, θ::AbstractVecOrMat, r::AbstractVecOrMat) = ∂H∂r(h, r)

### hamiltonian.jl

import AdvancedHMC: refresh, phasepoint
using AdvancedHMC: FullMomentumRefreshment, PartialMomentumRefreshment, AbstractMetric

# To change L180 of hamiltonian.jl
phasepoint(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    θ::AbstractVecOrMat{T},
    h::Hamiltonian
) where {T<:Real} = phasepoint(h, θ, rand(rng, h.metric, h.kinetic, θ))

# To change L191 of hamiltonian.jl
refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::FullMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
) = phasepoint(h, z.θ, rand(rng, h.metric, h.kinetic, z.θ))

# To change L215 of hamiltonian.jl
refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ref::PartialMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
) = phasepoint(h, z.θ, ref.α * z.r + sqrt(1 - ref.α^2) * rand(rng, h.metric, h.kinetic, z.θ))

# To change L146 of metric.jl
# Ignore θ by default (i.e. not position-dependent)
Base.rand(rng::AbstractRNG, metric::AbstractMetric, kinetic, θ) = _rand(rng, metric, kinetic)    # this disambiguity is required by Random.rand
Base.rand(rng::AbstractVector{<:AbstractRNG}, metric::AbstractMetric, kinetic, θ) = _rand(rng, metric, kinetic)
Base.rand(metric::AbstractMetric, kinetic, θ) = rand(GLOBAL_RNG, metric, kinetic)

### metric.jl

import AdvancedHMC: _rand
using AdvancedHMC: AbstractMetric
using LinearAlgebra: eigen, cholesky, Symmetric

abstract type AbstractRiemannianMetric <: AbstractMetric end

abstract type AbstractHessianMap end

struct IdentityMap <: AbstractHessianMap end

(::IdentityMap)(x) = x

struct SoftAbsMap{T} <: AbstractHessianMap 
    α::T
end

# TODO Register softabs with ReverseDiff
#! The definition of SoftAbs from Page 3 of Betancourt (2012)
function softabs(X, α=20.0)
    F = eigen(X) # ReverseDiff cannot diff through `eigen`
    Q = hcat(F.vectors)
    λ = F.values
    softabsλ = λ .* coth.(α * λ)
    return Q * diagm(softabsλ) * Q', Q, λ, softabsλ
end

(map::SoftAbsMap)(x) = softabs(x, map.α)[1]

struct DenseRiemannianMetric{
    T,
    TM<:AbstractHessianMap,
    A<:Union{Tuple{Int},Tuple{Int,Int}},
    AV<:AbstractVecOrMat{T},
    TG,
    T∂G∂θ,
} <: AbstractRiemannianMetric
    size::A
    G::TG # TODO store G⁻¹ here instead
    ∂G∂θ::T∂G∂θ
    map::TM
    _temp::AV
end

# TODO Make dense mass matrix support matrix-mode parallel
function DenseRiemannianMetric(size, G, ∂G∂θ, map=IdentityMap()) where {T<:AbstractFloat}
    _temp = Vector{Float64}(undef, size[1])
    return DenseRiemannianMetric(size, G, ∂G∂θ, map, _temp)
end
# DenseEuclideanMetric(::Type{T}, D::Int) where {T} = DenseEuclideanMetric(Matrix{T}(I, D, D))
# DenseEuclideanMetric(D::Int) = DenseEuclideanMetric(Float64, D)
# DenseEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T} = DenseEuclideanMetric(Matrix{T}(I, first(sz), first(sz)))
# DenseEuclideanMetric(sz::Tuple{Int}) = DenseEuclideanMetric(Float64, sz)

# renew(ue::DenseEuclideanMetric, M⁻¹) = DenseEuclideanMetric(M⁻¹)

Base.size(e::DenseRiemannianMetric) = e.size
Base.size(e::DenseRiemannianMetric, dim::Int) = e.size[dim]
Base.show(io::IO, dem::DenseRiemannianMetric) = print(io, "DenseRiemannianMetric(...)")

function _rand(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    metric::DenseRiemannianMetric{T},
    kinetic,
    θ,
) where {T}
    r = randn(rng, T, size(metric)...)
    G⁻¹ = inv(metric.map(metric.G(θ)))
    chol = cholesky(Symmetric(G⁻¹))
    ldiv!(chol.U, r)
    return r
end

Base.rand(rng::AbstractRNG, metric::AbstractRiemannianMetric, kinetic, θ) = _rand(rng, metric, kinetic, θ)
Base.rand(rng::AbstractVector{<:AbstractRNG}, metric::AbstractRiemannianMetric, kinetic, θ) = _rand(rng, metric, kinetic, θ)

### hamiltonian.jl

import AdvancedHMC: phasepoint, neg_energy, ∂H∂θ, ∂H∂r
using LinearAlgebra: logabsdet, tr

# QUES Do we want to change everything to position dependent by default?
# Add θ to ∂H∂r for DenseRiemannianMetric
phasepoint(
    h::Hamiltonian{<:DenseRiemannianMetric},
    θ::T,
    r::T;
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, r, θ))
) where {T<:AbstractVecOrMat} = PhasePoint(θ, r, ℓπ, ℓκ)

# Negative kinetic energy
#! Eq (13) of Girolami & Calderhead (2011)
function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric},
    r::T,
    θ::T
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    D = size(G, 1)
    # Need to consider the normalizing term as it is no longer same for different θs
    lad, s = logabsdet(G)
    # NOTE Removed per Hong's suggestion as it's not the correct way to handle numeric issue here
    # if s == -1
    #     return Inf # trigger numeric error to reject the current position
    #                 # QUES Is this a valid work-around?
    # end
    logZ = 1 / 2 * (D * log(2π) + lad * s)
    mul!(h.metric._temp, inv(G), r)
    return -logZ - dot(r, h.metric._temp) / 2
end

# QUES L31 of hamiltonian.jl now reads a bit weird (semantically)
function ∂H∂θ(h::Hamiltonian{<:DenseRiemannianMetric{T,<:IdentityMap}}, θ::AbstractVecOrMat{T}, r::AbstractVecOrMat{T}) where {T}
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
    G = h.metric.map(h.metric.G(θ))
    invG = inv(G)
    ∂G∂θ = h.metric.∂G∂θ(θ)
    
    d = length(∂ℓπ∂θ)
    return DualValue(
        ℓπ, 
        #! Eq (15) of Girolami & Calderhead (2011)
        -mapreduce(vcat, 1:d) do i
            ∂G∂θᵢ = ∂G∂θ[:,:,i]
            #! Looks like the first negative sign is a typo in (15)
            # ∂ℓπ∂θ[i] - 1 / 2 * tr(invG * ∂G∂θᵢ) + 1 / 2 * r' * invG * ∂G∂θᵢ * invG * r
            ∂ℓπ∂θ[i] + 1 / 2 * tr(invG * ∂G∂θᵢ) + 1 / 2 * r' * invG * ∂G∂θᵢ * invG * r
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
        J[i,j] = (λ[i] == λ[j]) ? 
            dsoftabsdλ(α, λ[i]) : 
            ((λ[i] * coth(α * λ[i]) - λ[j] * coth(α * λ[j])) / (λ[i] - λ[j]))
    end
    return J
end

function ∂H∂θ(h::Hamiltonian{<:DenseRiemannianMetric{T, <:SoftAbsMap}}, θ::AbstractVecOrMat{T}, r::AbstractVecOrMat{T}) where {T}
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
    H = h.metric.G(θ)
    # println("H: ", H)
    G, Q, λ, softabsλ = softabs(H, h.metric.map.α)
    # println("Q: ", Q)
    softabsΛ = diagm(softabsλ)
    R = diagm(1 ./ softabsλ)
    # println("R: ", R)
    # M = inv(softabsΛ) * Q' * r
    M = R * Q' * r
    D = diagm((Q' * r) ./ softabsλ)
    # println("M: ", M)
    invG = inv(G)
    ∂H∂θ = h.metric.∂G∂θ(θ)
    # println("∂H∂θ: ", ∂H∂θ)
    J = make_J(λ, h.metric.map.α)
    # println("J: ", J)
    
    d = length(∂ℓπ∂θ)
    #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
    g = -mapreduce(vcat, 1:d) do i
        ∂H∂θᵢ = ∂H∂θ[:,:,i]
        # ∂ℓπ∂θ[i] - 1 / 2 * tr(Q * (R .* J) * Q' * ∂H∂θᵢ) + 1 / 2 * M' * (J .* Q' * ∂H∂θᵢ * Q) * M
        ∂ℓπ∂θ[i] - 1 / 2 * tr(Q * (R .* J) * Q' * ∂H∂θᵢ) + 1 / 2 * tr(Q * D * J * D * Q' * ∂H∂θᵢ)
        # -1 / 2 * tr(Q * (R .* J) * Q' * ∂H∂θᵢ) # first term checks out
        # TODO Figure out why v1 doesn't work
        # +1 / 2 * M' * (J .* Q' * ∂H∂θᵢ * Q) * M # second term (v1)
        # +1 / 2 * tr(Q * D * J * D * Q' * ∂H∂θᵢ) # second term (v2) checks out
    end
    # println("g: ", g)
    return DualValue(
        ℓπ, 
        g,
    )
end

#! Eq (14) of Girolami & Calderhead (2011)
function ∂H∂r(h::Hamiltonian{<:DenseRiemannianMetric}, θ::AbstractVecOrMat, r::AbstractVecOrMat)
    H = h.metric.G(θ)
    # if any(.!(isfinite.(H)))
    #     println("θ: ", θ)
    #     println("H: ", H)
    # end
    G = h.metric.map(H)
    # return inv(G) * r
    # println("G \ r: ", G \ r)
    return G \ r # NOTE it's actually pretty weird that ∂H∂θ returns DualValue but ∂H∂r doesn't
end
