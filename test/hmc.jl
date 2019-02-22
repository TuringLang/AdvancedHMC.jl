using Test
using LinearAlgebra: dot
using Statistics: mean
using Distributions, ForwardDiff, HMC

const d = 5
_logπ(θ::AbstractVector{T}) where {T<:Real} = logpdf(MvNormal(zeros(d), ones(d)), θ)

_dlogπdθ = θ -> ForwardDiff.gradient(_logπ, θ)

h = Hamiltonian(UnitMetric(), _logπ, _dlogπdθ)
ϵ = 0.01
t = StaticTrajectory(LastFromTraj(), Leapfrog(ϵ), 10)
θ = randn(d)
samples = HMC.sample(h, t, θ, 10_000)

println(mean(samples))
