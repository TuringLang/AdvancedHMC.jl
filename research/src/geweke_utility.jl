using Distributions, DynamicPPL
using VecTargets: ContinuousMultivariateDistribution, _logpdf_normal_std
import VecTargets: dim, logpdf



struct SimpleTarget <: ContinuousMultivariateDistribution end

dim(st::SimpleTarget) = 3

function _logpdf_st(θx::AbstractVecOrMat)
    θ1, θ2, x = θx[1,:], θx[2,:], θx[3,:]
    s = 1

    lp11 = _logpdf_normal_std(θ1, 0, s) # _logpdf_normal_std(x, m, s)
    lp12 = _logpdf_normal_std(θ2, 0, s)
    lp2 = _logpdf_normal_std(x, θ1, s)
    
    return lp11 + lp12 + lp2
end

logpdf(::SimpleTarget, θ::AbstractVector) = only(_logpdf_st(θ))

logpdf(::SimpleTarget, θ::AbstractMatrix) = _logpdf_st(θ)

# Treating the first 2 dimensions as latent and the third as data.
@model function TuringSimple(θ=missing, x=missing)
    if ismissing(θ)
        θ = Vector(undef, 2)
    end
    θ[1] ~ Normal(0, 1)
    θ[2] ~ Normal(0, 1)
    x ~ Normal(θ[1], 1)
    return θ, x
end



# Treating the first 2 dimensions as latent and the third as data.
@model function TuringFunnel(θ=missing, x=missing)
    if ismissing(θ)
        θ = Vector(undef, 2)
    end
    θ[1] ~ Normal(0, 3)
    s = exp(θ[1] / 2)
    θ[2] ~ Normal(0, s)
    x ~ Normal(0, s)
    return θ, x
end