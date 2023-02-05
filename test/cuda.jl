using ReTest
using AdvancedHMC
using AdvancedHMC: DualValue, PhasePoint
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

@testset "PhasePoint GPU" begin
    for T in [Float32, Float64]
        init_z1() = PhasePoint(
            CuArray([T(NaN) T(NaN)]),
            CuArray([T(NaN) T(NaN)]),
            DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
            DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
        )
        init_z2() = PhasePoint(
            CuArray([T(Inf) T(Inf)]),
            CuArray([T(Inf) T(Inf)]),
            DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
            DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
        )

        @test_logs (
            :warn,
            "The current proposal will be rejected due to numerical error(s).",
        ) init_z1()
        @test_logs (
            :warn,
            "The current proposal will be rejected due to numerical error(s).",
        ) init_z2()

        z1 = init_z1()
        z2 = init_z2()

        @test z1.ℓπ.value == z2.ℓπ.value
        @test z1.ℓκ.value == z2.ℓκ.value
    end
end
