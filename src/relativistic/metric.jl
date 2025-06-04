function rand_angles(rng::AbstractRNG, dim)
    return rand(rng, dim - 1) .* vcat(fill(π, dim - 2), 2 * π)
end

"Special case of `polar2spherical` with dimension equal 2"
polar2cartesian(θ, d) = d * [cos(θ), sin(θ)]

# ref: https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
function polar2spherical(θs, d)
    cos_lst, sin_lst = cos.(θs), sin.(θs)
    suffixed_cos_lst = vcat(cos_lst, 1) # [cos(θ[1]), cos(θ[2]), ..., cos(θ[d-1]), 1]
    prefixed_cumprod_sin_lst = vcat(1, cumprod(sin_lst)) # [1, sin(θ[1]), sin(θ[1]) * sin(θ[2]), ..., sin(θ[1]) * ... * sin(θ[d-1])]
    return d * prefixed_cumprod_sin_lst .* suffixed_cos_lst
end

momentum_mode(m, c) = sqrt((1 / c^2 + sqrt(1 / c^2 + 4 * m^2)) / 2) # mode of the momentum distribution

function rand_momentum(
    rng::AbstractRNG,
    metric::UnitEuclideanMetric{T},
    kinetic::RelativisticKinetic{T},
    ::AbstractVecOrMat,
) where {T}
    densityfunc = x -> exp(-relativistic_energy(kinetic, [x])) * x
    mm = momentum_mode(kinetic.m, kinetic.c)
    sampler = RejectionSampler(densityfunc, (0.0, Inf), (mm / 2, mm * 2); max_segments=5)
    sz = size(metric)
    θs = rand_angles(rng, prod(sz))
    d = only(run_sampler!(rng, sampler, 1))
    r = polar2spherical(θs, d * rand(rng, [-1, +1])) # TODO Double check if +/- is needed
    r = reshape(r, sz)
    return r
end

# TODO Support AbstractVector{<:AbstractRNG}
# FIXME Unit-test this using slice sampler or HMC sampler
function rand_momentum(
    rng::AbstractRNG,
    metric::UnitEuclideanMetric{T},
    kinetic::DimensionwiseRelativisticKinetic{T},
    ::AbstractVecOrMat,
) where {T}
    h_temp = Hamiltonian(metric, kinetic, identity, identity)
    densityfunc = x -> exp(neg_energy(h_temp, [x], [x]))
    sampler = RejectionSampler(densityfunc, (-Inf, Inf); max_segments=5)
    sz = size(metric)
    r = run_sampler!(rng, sampler, prod(sz))
    r = reshape(r, sz)
    return r
end

# TODO Support AbstractVector{<:AbstractRNG}
function rand_momentum(
    rng::AbstractRNG,
    metric::DiagEuclideanMetric{T},
    kinetic::AbstractRelativisticKinetic{T},
    θ::AbstractVecOrMat,
) where {T}
    r = rand_momentum(rng, UnitEuclideanMetric(size(metric)), kinetic, θ)
    # p' = A p where A = sqrtM
    r ./= metric.sqrtM⁻¹
    return r
end
# TODO Support AbstractVector{<:AbstractRNG}
function rand_momentum(
    rng::AbstractRNG,
    metric::DenseEuclideanMetric{T},
    kinetic::AbstractRelativisticKinetic{T},
    θ::AbstractVecOrMat,
) where {T}
    r = rand_momentum(rng, UnitEuclideanMetric(size(metric)), kinetic, θ)
    # p' = A p where A = cholM
    ldiv!(metric.cholM⁻¹, r)
    return r
end
