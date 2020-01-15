# TODO: add more target distributions and make them iteratable

# Dimension of testing distribution
const D = 5
# Tolerance ratio
const TRATIO = Int == Int64 ? 1 : 2
# Deterministic tolerance
const DETATOL = 1e-3 * D * TRATIO
# Random tolerance
const RNDATOL = 5e-2 * D * TRATIO

# Hand-coded multivaite Gaussain

const gaussian_m = zeros(D)
const gaussian_s = ones(D)

function ℓπ(m, s, x::AbstractVecOrMat{T}) where {T}
    diff = x .- m
    v = s.^2
    return -(log(2 * T(pi)) .+ log.(v) .+ diff .* diff ./ v) / 2
end

function ℓπ(θ::AbstractVector)
    return sum(ℓπ(gaussian_m, gaussian_s, θ))
end

function ℓπ(θ::AbstractMatrix)
    return dropdims(sum(ℓπ(gaussian_m, gaussian_s, θ); dims=1); dims=1)
end

function ∂ℓπ∂θ(m, s, x::AbstractVecOrMat{T}) where {T}
    diff = x .- m
    v = s.^2
    v = -(log(2 * T(pi)) .+ log.(v) .+ diff .* diff ./ v) / 2
    g = -diff
    return v, g
end

function ∂ℓπ∂θ(θ::AbstractVector)
    v, g = ∂ℓπ∂θ(gaussian_m, gaussian_s, θ)
    return sum(v), g
end

function ∂ℓπ∂θ(θ::AbstractMatrix)
    v, g = ∂ℓπ∂θ(gaussian_m, gaussian_s, θ)
    return dropdims(sum(v; dims=1); dims=1), g
end

# For the Turing model
# @model gdemo() = begin
#     s ~ InverseGamma(2, 3)
#     m ~ Normal(0, sqrt(s))
#     1.5 ~ Normal(m, sqrt(s))
#     2.0 ~ Normal(m, sqrt(s))
#     return s, m
# end

using Distributions: logpdf, InverseGamma, Normal
using Bijectors: invlink, logpdf_with_trans

function invlink_gdemo(θ)
    s = invlink(InverseGamma(2, 3), θ[1])
    m = θ[2]
    return [s, m]
end

function ℓπ_gdemo(θ)
    s, m = invlink_gdemo(θ)
    logprior = logpdf_with_trans(InverseGamma(2, 3), s, true) + logpdf(Normal(0, sqrt(s)), m)
    loglikelihood = logpdf(Normal(m, sqrt(s)), 1.5) + logpdf(Normal(m, sqrt(s)), 2.0)
    return logprior + loglikelihood
end
