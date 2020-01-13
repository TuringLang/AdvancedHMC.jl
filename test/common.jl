# TODO: add more target distributions and make them iteratable

# Dimension of testing distribution
const D = 5
# Tolerance ratio
const TRATIO = Int == Int64 ? 1 : 2
# Deterministic tolerance
const DETATOL = 1e-3 * D * TRATIO
# Random tolerance
const RNDATOL = 5e-2 * D * TRATIO

using Distributions: logpdf, MvNormal, InverseGamma, Normal
using Bijectors: link, invlink, logpdf_with_trans

const dm = zeros(D)
const dσ = ones(D)
const MVN = MvNormal(dm, dσ)
ℓπ(θ) = logpdf(MVN, θ)

# Manual implementation of gradient so that it support matrix mode
function ∂ℓπ∂θ(θ::AbstractVecOrMat)
    d = MVN
    diff = θ .- dm
    dims = θ isa AbstractVector ? (:) : 1
    v = -(D * log(2π) + 2 * sum(log.(dσ)) .+ sum(abs2, diff ./ dσ; dims=dims)) / 2
    v = θ isa AbstractMatrix ? vec(v) : v
    g = -diff
    return (v, g)
end

# For the Turing model
# @model gdemo() = begin
#     s ~ InverseGamma(2, 3)
#     m ~ Normal(0, sqrt(s))
#     1.5 ~ Normal(m, sqrt(s))
#     2.0 ~ Normal(m, sqrt(s))
#     return s, m
# end

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
