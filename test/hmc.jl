using Test
using LinearAlgebra: dot
using Statistics: mean
using ForwardDiff, HMC

function _π(θ::AbstractVector{T}) where {T<:Real}
  d = ones(length(θ)) * 3
  return exp(-(dot(θ, θ))) + exp(-(dot(θ - d, θ - d))) + exp(-(dot(θ + d, θ + d)))
end

_logπ(θ::AbstractVector{T}) where {T<:Real} = log(_π(θ))

_dlogπdθ = θ -> ForwardDiff.gradient(_logπ, θ)

h = Hamiltonian(UnitMetric(), _logπ, _dlogπdθ)
ϵ = 0.1
t = StaticTrajectory(LastFromTraj(), Leapfrog(ϵ), 10)
θ = randn(3)
samples = sample(h, t, θ, 10_000)

println(mean(samples))
