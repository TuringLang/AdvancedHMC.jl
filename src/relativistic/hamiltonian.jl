abstract type AbstractRelativisticKinetic{T} <: AbstractKinetic end

struct RelativisticKinetic{T} <: AbstractRelativisticKinetic{T}
    "Mass"
    m::T
    "Speed of light"
    c::T
end

relativistic_mass(kinetic::RelativisticKinetic, r, r′ = r) =
    kinetic.m * sqrt(dot(r, r′) / (kinetic.m^2 * kinetic.c^2) + 1)
relativistic_energy(kinetic::RelativisticKinetic, r, r′ = r) =
    sum(kinetic.c^2 * relativistic_mass(kinetic, r, r′))

struct DimensionwiseRelativisticKinetic{T} <: AbstractRelativisticKinetic{T}
    "Mass"
    m::T
    "Speed of light"
    c::T
end

relativistic_mass(kinetic::DimensionwiseRelativisticKinetic, r, r′ = r) =
    kinetic.m .* sqrt.(r .* r′ ./ (kinetic.m .^ 2 .* kinetic.c .^ 2) .+ 1)
relativistic_energy(kinetic::DimensionwiseRelativisticKinetic, r, r′ = r) =
    sum(kinetic.c .^ 2 .* relativistic_mass(kinetic, r, r′))

function ∂H∂r(
    h::Hamiltonian{<:UnitEuclideanMetric,<:AbstractRelativisticKinetic},
    r::AbstractVecOrMat,
)
    mass = relativistic_mass(h.kinetic, r)
    return r ./ mass
end
function ∂H∂r(
    h::Hamiltonian{<:DiagEuclideanMetric,<:AbstractRelativisticKinetic},
    r::AbstractVecOrMat,
)
    r = h.metric.sqrtM⁻¹ .* r
    mass = relativistic_mass(h.kinetic, r)
    red_term = r ./ mass # red part of (15)
    return h.metric.sqrtM⁻¹ .* red_term # (15)
end
function ∂H∂r(
    h::Hamiltonian{<:DenseEuclideanMetric,<:AbstractRelativisticKinetic},
    r::AbstractVecOrMat,
)
    r = h.metric.cholM⁻¹ * r
    mass = relativistic_mass(h.kinetic, r)
    red_term = r ./ mass
    return h.metric.cholM⁻¹' * red_term
end

function neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric,<:AbstractRelativisticKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVector}
    return -relativistic_energy(h.kinetic, r)
end
function neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric,<:AbstractRelativisticKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVector}
    r = h.metric.sqrtM⁻¹ .* r
    return -relativistic_energy(h.kinetic, r)
end
function neg_energy(
    h::Hamiltonian{<:DenseEuclideanMetric,<:AbstractRelativisticKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVector}
    r = h.metric.cholM⁻¹ * r
    return -relativistic_energy(h.kinetic, r)
end