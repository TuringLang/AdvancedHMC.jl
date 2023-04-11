using AdvancedHMC
import AdvancedHMC: ∂H∂r, neg_energy, AbstractKinetic
import Random: AbstractRNG

struct RelativisticKinetic{T} <: AbstractKinetic
    "Mass"
    m::T
    "Speed of light"
    c::T
end


function ∂H∂r(
    h::Hamiltonian{<:UnitEuclideanMetric,<:RelativisticKinetic},
    r::AbstractVecOrMat,
)
    mass = h.kinetic.m .* sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 * h.kinetic.c .^ 2) .+ 1)
    return r ./ mass
end
function ∂H∂r(
    h::Hamiltonian{<:DiagEuclideanMetric,<:RelativisticKinetic},
    r::AbstractVecOrMat,
)
    r = h.metric.sqrtM⁻¹ .* r
    mass = h.kinetic.m .* sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 * h.kinetic.c .^ 2) .+ 1)
    retval = r ./ mass # red part of (15)
    return h.metric.sqrtM⁻¹ .* retval # (15)
end


function neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric,<:RelativisticKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVector}
    return -sum(
        h.kinetic.m .* h.kinetic.c .^ 2 .*
        sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 .* h.kinetic.c .^ 2) .+ 1),
    )
end

function neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric,<:RelativisticKinetic},
    r::T,
    θ::T,
) where {T<:AbstractVector}
    r = h.metric.sqrtM⁻¹ .* r
    return -sum(
        h.kinetic.m .* h.kinetic.c .^ 2 .*
        sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 .* h.kinetic.c .^ 2) .+ 1),
    )
end


using AdaptiveRejectionSampling: RejectionSampler, run_sampler!
import AdvancedHMC: _rand

# TODO Support AbstractVector{<:AbstractRNG}
function _rand(
    rng::AbstractRNG,
    metric::UnitEuclideanMetric{T},
    kinetic::RelativisticKinetic{T},
) where {T}
    h_temp = Hamiltonian(metric, kinetic, identity, identity)
    densityfunc = x -> exp(neg_energy(h_temp, [x], [x]))
    sampler = RejectionSampler(densityfunc, (-Inf, Inf); max_segments = 5)
    sz = size(metric)
    r = run_sampler!(rng, sampler, prod(sz))
    r = reshape(r, sz)
    return r
end

# TODO Support AbstractVector{<:AbstractRNG}
function _rand(
    rng::AbstractRNG,
    metric::DiagEuclideanMetric{T},
    kinetic::RelativisticKinetic{T},
) where {T}
    r = _rand(rng, UnitEuclideanMetric(size(metric)), kinetic)
    # p' = A p where A = sqrtM
    r ./= metric.sqrtM⁻¹
    return r
end
