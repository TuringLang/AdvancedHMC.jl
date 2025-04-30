using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", ".."))

using Test
using AdvancedHMC
using AdvancedHMC: DualValue, PhasePoint
using CUDA
using LogDensityProblems

include(joinpath(@__DIR__, "..", "common.jl"))

@testset "AdvancedHMC GPU" begin
    if CUDA.functional()
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
    else
        println("GPU tests are skipped because no CUDA devices are found.")
    end
end

@testset "PhasePoint GPU" begin
    if CUDA.functional()
        for T in [Float32, Float64]
            function init_z1()
                return PhasePoint(
                    CuArray([T(NaN) T(NaN)]),
                    CuArray([T(NaN) T(NaN)]),
                    DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
                    DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
                )
            end
            function init_z2()
                return PhasePoint(
                    CuArray([T(Inf) T(Inf)]),
                    CuArray([T(Inf) T(Inf)]),
                    DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
                    DualValue(CuArray(zeros(T, 2)), CuArray(zeros(T, 1, 2))),
                )
            end
    
            z1 = init_z1()
            z2 = init_z2()
    
            @test z1.ℓπ.value == z2.ℓπ.value
            @test z1.ℓκ.value == z2.ℓκ.value
        end
    else
        println("GPU tests are skipped because no CUDA devices are found.")
    end
end
