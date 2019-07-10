# TODO: add more target distributions and make them iteratable

# Dimension of testing distribution
const D = 5

# Deterministic tolerance
const DETATOL = 1e-3 * D
# Random tolerance
const RNDATOL = 5e-2 * D

using Distributions: logpdf, MvNormal

ℓπ(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)

using DiffResults: GradientResult, value, gradient
using ForwardDiff: gradient!

function ∂ℓπ∂θ(θ)
    res = GradientResult(θ)
    gradient!(res, ℓπ, θ)
    return (value(res), gradient(res))
end
