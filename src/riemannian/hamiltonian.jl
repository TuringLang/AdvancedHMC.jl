# QUES Do we want to change everything to position dependent by default?
# Add θ to ∂H∂r for DenseRiemannianMetric
phasepoint(
    h::Hamiltonian{<:DenseRiemannianMetric},
    θ::T,
    r::T;
    ℓπ = ∂H∂θ(h, θ),
    ℓκ = DualValue(neg_energy(h, r, θ), ∂H∂r(h, θ, r)),
) where {T<:AbstractVecOrMat} = PhasePoint(θ, r, ℓπ, ℓκ)

# Negative kinetic energy
#! Eq (13) of Girolami & Calderhead (2011)
function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric,<:GaussianKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    # Need to consider the normalizing term as it is no longer same for different θs
    logZ = 1 / 2 * (size(G, 1) * log(2π) + logdet(G)) # it will be user's responsibility to make sure G is SPD and logdet(G) is defined
    mul!(h.metric._temp, inv(G), r)
    # ldiv!(h.metric._temp, cholesky(G), r) # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ldiv!
    return -logZ - dot(r, h.metric._temp) / 2
end

function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric,<:AbstractRelativisticKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    # Need to consider the normalizing term as it is no longer same for different θs
    logZ_partial = 1 / 2 * logdet(G)
    # M⁻¹ = inv(G)
    # cholM⁻¹ = cholesky(Symmetric(M⁻¹)).U
    # r = cholM⁻¹ * r
    # return -relativistic_energy(h.kinetic, r) - logZ_partial 

    mul!(h.metric._temp, inv(G), r)
    # ldiv!(h.metric._temp, cholesky(G), r) # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ldiv!
    return -relativistic_energy(h.kinetic, r, h.metric._temp) - logZ_partial
end

# QUES L31 of hamiltonian.jl now reads a bit weird (semantically)
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

function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:IdentityMap},<:AbstractRelativisticKinetic},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    error("unimplemented")
end

# Ref: https://www.wolframalpha.com/input?i=derivative+of+x+*+coth%28a+*+x%29
#! Based on middle of the right column of Page 3 of Betancourt (2012) "Note that when λi=λj, such as for the diagonal elementsor degenerate eigenvalues, this becomes the derivative"
dsoftabsdλ(α, λ) = coth(α * λ) + λ * α * -csch(λ * α)^2

#! J as defined in middle of the right column of Page 3 of Betancourt (2012)
function make_J(λ::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    d = length(λ)
    J = Matrix{T}(undef, d, d)
    for i = 1:d, j = 1:d
        J[i, j] =
            (λ[i] == λ[j]) ? dsoftabsdλ(α, λ[i]) :
            ((λ[i] * coth(α * λ[i]) - λ[j] * coth(α * λ[j])) / (λ[i] - λ[j]))
    end
    return J
end

# TODO Add stricter types to block hamiltonian.jl#L37 from working on unknown metric/kinetic
∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap},<:GaussianKinetic},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T} = ∂H∂θ_cache(h, θ, r)
function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap},<:GaussianKinetic},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache = false,
    cache = nothing,
) where {T}
    # Terms that only dependent on θ can be cached in θ-unchanged loops
    if isnothing(cache)
        ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
        H = h.metric.G(θ)
        ∂H∂θ = h.metric.∂G∂θ(θ)

        G, Q, λ, softabsλ = softabs(H, h.metric.map.α)

        R = Diagonal(1 ./ softabsλ)

        # softabsΛ = Diagonal(softabsλ)
        # M = inv(softabsΛ) * Q' * r
        # M = R * Q' * r # equiv to above but avoid inv

        J = make_J(λ, h.metric.map.α)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        term_1_cached = Q * (R .* J) * Q'
    else
        (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached) = cache
    end
    d = length(∂ℓπ∂θ)
    D = Diagonal((Q' * r) ./ softabsλ)
    term_2 = Q * D * J * D * Q'

    isdiag = ∂H∂θ isa AbstractMatrix
    g =
        isdiag ?
        -(∂ℓπ∂θ - 1 / 2 * diag(term_1_cached * ∂H∂θ) + 1 / 2 * diag(term_2 * ∂H∂θ)) :
        -mapreduce(vcat, 1:d) do i
            ∂H∂θᵢ = ∂H∂θ[:, :, i]
            # ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * M' * (J .* (Q' * ∂H∂θᵢ * Q)) * M # (v1)
            # NOTE Some further optimization can be done here: cache the 1st product all together
            ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * tr(term_2 * ∂H∂θᵢ) # (v2) cache friendly
        end

    dv = DualValue(ℓπ, g)
    return return_cache ? (dv, (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached)) : dv
end

∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap},<:RelativisticKinetic},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T} = ∂H∂θ_cache(h, θ, r)
function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{T,<:SoftAbsMap},<:RelativisticKinetic},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache = false,
    cache = nothing,
) where {T}
    # Terms that only dependent on θ can be cached in θ-unchanged loops
    if isnothing(cache)
        ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
        H = h.metric.G(θ)
        ∂H∂θ = h.metric.∂G∂θ(θ)

        G, Q, λ, softabsλ = softabs(H, h.metric.map.α)

        R = Diagonal(1 ./ softabsλ)

        # softabsΛ = Diagonal(softabsλ)
        # M = inv(softabsΛ) * Q' * r
        # M = R * Q' * r # equiv to above but avoid inv

        J = make_J(λ, h.metric.map.α)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        term_1_cached = Q * (R .* J) * Q'

        M = G
    else
        (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached, M) = cache
    end
    d = length(∂ℓπ∂θ)
    D = Diagonal((Q' * r) ./ softabsλ)
    term_2 = Q * D * J * D * Q'

    #! (18) of Overleaf note
    mass = relativistic_mass(h.kinetic, r, M \ r)

    # TODO convert the code below to a test case
    # @info diag(term_1_cached * ∂H∂θ)
    # function _Hdiag_transform(Hdiag)
    #     _T = eltype(Hdiag)
    #     _, d = size(Hdiag)
    #     return mapreduce(vcat, enumerate(eachrow(Hdiag))) do (i, row)
    #         top = zeros(_T, i - 1, d)
    #         bot = zeros(_T, d - i, d)
    #         vcat(top, row', bot)
    #     end
    # end
    # function reshape_∂G∂θ(H)
    #     d = size(H, 2)
    #     return cat((H[(i-1)*d+1:i*d, :] for i = 1:d)...; dims = 3)
    # end
    # ∂H∂θ = reshape_∂G∂θ(_Hdiag_transform(∂H∂θ))
    # @info mapreduce(vcat, 1:d) do i
    #     tr(term_1_cached * ∂H∂θ[:,:,i])
    # end
    # @assert false

    isdiag = ∂H∂θ isa AbstractMatrix
    g =
        isdiag ?
        -(
            ∂ℓπ∂θ - 1 / 2 * diag(term_1_cached * ∂H∂θ) +
            1 / 2 * (1 / mass) * diag(term_2 * ∂H∂θ)
        ) :
        -mapreduce(vcat, 1:d) do i
            ∂H∂θᵢ = ∂H∂θ[:, :, i]
            ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) +
            1 / 2 * (1 / mass) * tr(term_2 * ∂H∂θᵢ) # (v2) cache friendly
        end

    dv = DualValue(ℓπ, g)
    return return_cache ? (dv, (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached, M)) : dv
end

# FIXME This implementation for dimensionwise is incorrect
function ∂H∂θ(
    h::Hamiltonian{
        <:DenseRiemannianMetric{T,<:SoftAbsMap},
        <:DimensionwiseRelativisticKinetic,
    },
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
    H = h.metric.G(θ)
    ∂H∂θ = h.metric.∂G∂θ(θ)

    G, Q, λ, softabsλ = softabs(H, h.metric.map.α)

    R = Diagonal(1 ./ softabsλ)

    # softabsΛ = Diagonal(softabsλ)
    # M = inv(softabsΛ) * Q' * r
    # M = R * Q' * r # equiv to above but avoid inv

    J = make_J(λ, h.metric.map.α)


    d = length(∂ℓπ∂θ)
    D = Diagonal((Q' * r) ./ softabsλ)

    #! (18) of Overleaf note
    M⁻¹ = inv(G)
    cholM⁻¹ = cholesky(Symmetric(M⁻¹)).U
    r = cholM⁻¹ * r
    mass = relativistic_mass(h.kinetic, r)
    term_1_cached = 1 ./ mass

    term_2_cached = Q * D * J * D * Q'
    g = -mapreduce(vcat, 1:d) do i
        ∂H∂θᵢ = ∂H∂θ[:, :, i]
        # ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * M' * (J .* (Q' * ∂H∂θᵢ * Q)) * M # (v1)
        # NOTE Some further optimization can be done here: cache the 1st product all together
        ∂ℓπ∂θ[i] - 1 / 2 * term_1_cached[i] * -tr(term_2_cached * ∂H∂θᵢ) # (v2) cache friendly
    end

    return DualValue(ℓπ, g)
end

#! Eq (14) of Girolami & Calderhead (2011)
function ∂H∂r(
    h::Hamiltonian{<:DenseRiemannianMetric,<:GaussianKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
)
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

function ∂H∂r(
    h::Hamiltonian{<:DenseRiemannianMetric,<:AbstractRelativisticKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
)
    M = h.metric.map(h.metric.G(θ))
    # M⁻¹ = inv(M)
    # cholM⁻¹ = cholesky(Symmetric(M⁻¹)).U

    # r = cholM⁻¹ * r
    # mass = relativistic_mass(h.kinetic, r)
    # red_term = r ./ mass
    # return cholM⁻¹' * red_term

    # Equivalent RelativisticKinetic implementation that avoid cholesky decomposition
    # M⁻¹r = M⁻¹ * r
    # mass = relativistic_mass(h.kinetic, r, M⁻¹r)
    # return M⁻¹r ./ mass

    # Equivalent RelativisticKinetic implementation that avoid matrix inversion and cholesky decomposition
    Mr = M \ r
    mass = relativistic_mass(h.kinetic, r, Mr)
    return Mr ./ mass
end