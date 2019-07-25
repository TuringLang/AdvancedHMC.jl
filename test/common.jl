# TODO: add more target distributions and make them iteratable

# Dimension of testing distribution
const D = 5

# Deterministic tolerance
const DETATOL = 1e-3 * D
# Random tolerance
const RNDATOL = 5e-2 * D

using Distributions: logpdf, MvNormal, InverseGamma, Normal
using DiffResults: GradientResult, value, gradient
using ForwardDiff: gradient!

ℓπ(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)

function ∂ℓπ∂θ(θ)
    res = GradientResult(θ)
    gradient!(res, ℓπ, θ)
    return (value(res), gradient(res))
end

function ℓπ_gdemo(θ)
    s = exp(θ[1])
    m = θ[2]
    logprior = logpdf(InverseGamma(2, 3), s) + log(s) + logpdf(Normal(0, sqrt(s)), m) 
    loglikelihood = logpdf(Normal(m, sqrt(s)), 1.5) + logpdf(Normal(m, sqrt(s)), 2.0)
    return logprior + loglikelihood
end

function ∂ℓπ∂θ_gdemo(θ)
    res = GradientResult(θ)
    gradient!(res, ℓπ_gdemo, θ)
    return (value(res), gradient(res))
end