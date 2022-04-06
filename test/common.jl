# TODO: add more target distributions and make them iteratable
# TODO: Integrate with https://github.com/xukai92/VecTargets.jl to achieve goal noted 
#       above.

# Dimension of testing distribution
const D = 5
# Tolerance ratio
const TRATIO = Int == Int64 ? 1 : 2
# Deterministic tolerance
const DETATOL = 1e-3 * D * TRATIO
# Random tolerance
const RNDATOL = 5e-2 * D * TRATIO * 2

# Hand-coded multivariate Gaussian

struct Gaussian{Tm, Ts}
    m::Tm
    s::Ts
end

function ℓπ_gaussian(g::AbstractVecOrMat{T}, s) where {T}
    return .-(log(2 * T(pi)) .+ 2 .* log.(s) .+ abs2.(g) ./ s.^2) ./ 2
end

ℓπ_gaussian(m, s, x) = ℓπ_gaussian(m .- x, s)

function ∇ℓπ_gaussianl(m, s, x)
    g = m .- x
    v = ℓπ_gaussian(g, s)
    return v, g
end

function get_ℓπ(g::Gaussian)
    ℓπ(x::AbstractVector) = sum(ℓπ_gaussian(g.m, g.s, x))
    ℓπ(x::AbstractMatrix) = dropdims(sum(ℓπ_gaussian(g.m, g.s, x); dims=1); dims=1)
    return ℓπ
end

function get_∇ℓπ(g::Gaussian)
    function ∇ℓπ(x::AbstractVector)
        val, grad = ∇ℓπ_gaussianl(g.m, g.s, x)
        return sum(val), grad
    end
    function ∇ℓπ(x::AbstractMatrix)
        val, grad = ∇ℓπ_gaussianl(g.m, g.s, x)
        return dropdims(sum(val; dims=1); dims=1), grad
    end
    return ∇ℓπ
end

ℓπ = get_ℓπ(Gaussian(zeros(D), ones(D)))
∂ℓπ∂θ = get_∇ℓπ(Gaussian(zeros(D), ones(D)))

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

test_show(x) = test_show(s -> length(s) > 0, x)
function test_show(pred, x)
    io = IOBuffer(; append = true)
    show(io, x)
    s = read(io, String)
    @test pred(s)
end
