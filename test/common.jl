# TODO: add more target distributions and make them iteratable

# Dimension of testing distribution
const D = 5
# Tolerance ratio
const TRATIO = Int == Int64 ? 1 : 2
# Deterministic tolerance
const DETATOL = 1e-3 * D * TRATIO
# Random tolerance
const RNDATOL = 5e-2 * D * TRATIO * 2

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

using Distributions: MvNormal
import Turing

Turing.@model function mvntest(θ=missing, x=missing)
    θ ~ MvNormal(zeros(D), 2)
    x ~ Normal(sum(θ), 1)
    return θ, x
end

function get_primitives(x, modelgen)
    spl_prior = Turing.SampleFromPrior()
    function ℓπ(θ)
        vi = Turing.VarInfo(model)
        vi[spl_prior] = θ
        model(vi, spl_prior)
        Turing.getlogp(vi)
    end
    adbackend = Turing.Core.ForwardDiffAD{40}
    alg_ad = Turing.HMC{adbackend}(0.1, 1)
    model = modelgen(missing, x)
    vi = Turing.VarInfo(model)
    spl = Turing.Sampler(alg_ad, model)
    Turing.Core.link!(vi, spl)
    ∂ℓπ∂θ = θ -> Turing.Core.gradient_logp(adbackend(), θ, vi, model, spl)
    θ₀ = Turing.VarInfo(model)[Turing.SampleFromPrior()]
    return ℓπ, ∂ℓπ∂θ, θ₀
end

function rand_θ_given(x, modelgen, metric, κ; n_samples=20)
    ℓπ, ∂ℓπ∂θ, θ₀ = get_primitives(x, modelgen)
    h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    samples, stats = sample(h, κ, θ₀, n_samples; verbose=false, progress=false)
    s = samples[end]
    return length(s) == 1 ? s[1] : s
end

# Test function
geweke_g(θ, x) = cat(θ, x; dims=1)
