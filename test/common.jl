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
using DiffResults: GradientResult, JacobianResult, value, gradient, jacobian
using ForwardDiff: gradient!, jacobian!
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

function ∂ℓπ∂θ_ad(θ::AbstractVector)
    res = GradientResult(θ)
    gradient!(res, ℓπ, θ)
    return (value(res), gradient(res))
end

function ∂ℓπ∂θ_ad(θ::AbstractMatrix)
    v = similar(θ, size(θ, 2))
    g = similar(θ)
    for i in 1:size(θ, 2)
        res = GradientResult(θ[:,i])
        gradient!(res, ℓπ, θ[:,i])
        v[i] = value(res)
        g[:,i] = gradient(res)
    end
    return (v, g)
end

function ∂ℓπ∂θ_viajacob(θ::AbstractMatrix)
    jacob = similar(θ)
    res = JacobianResult(similar(θ, size(θ, 2)), jacob)
    jacobian!(res, ℓπ, θ)
    jacob_full = jacobian(res)
    d, n = size(jacob)
    for i in 1:n
        jacob[:,i] = jacob_full[i,1+(i-1)*d:i*d]
    end
    return (value(res), jacob)
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

function ∂ℓπ∂θ_gdemo(θ)
    res = GradientResult(θ)
    gradient!(res, ℓπ_gdemo, θ)
    return (value(res), gradient(res))
end
