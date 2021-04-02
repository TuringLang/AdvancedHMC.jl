using Test
using AdvancedHMC
using CUDA

@testset "AdvancedHMC GPU" begin
    include("common.jl")

    n_chains = 1000
    n_samples = 1000
    dim = 5

    T = Float32
    m, s, θ₀ = zeros(T, dim), ones(T, dim), rand(T, dim, n_chains)
    m, s, θ₀ = CuArray(m), CuArray(s), CuArray(θ₀)

    target = Gaussian(m, s)
    metric = UnitEuclideanMetric(T, size(θ₀))
    ℓπ, ∇ℓπ = get_ℓπ(target), get_∇ℓπ(target)
    hamiltonian = Hamiltonian(metric, ℓπ, ∇ℓπ)
    integrator = Leapfrog(one(T) / 5)
    proposal = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(5)))

    samples, stats = sample(hamiltonian, proposal, θ₀, n_samples)
end