using Test, AdvancedHMC, LinearAlgebra, UnicodePlots, Random
using Statistics: mean, var, cov
include("common.jl")

@testset "Matrix mode" begin
    n_chains_max = 20
    n_chains_list = collect(1:n_chains_max)
    θ_init_list = [rand(D, n_chains) for n_chains in n_chains_list]
    ϵ = 0.2
    lf = Leapfrog(ϵ)
    i_test = 5
    lfi = Leapfrog(fill(ϵ, i_test))
    lfi_jittered = JitteredLeapfrog(fill(ϵ, i_test), 1.0)
    n_steps = 10
    n_samples = 12_000
    n_adapts = 2_000

    for metricT in [
        UnitEuclideanMetric,
        DiagEuclideanMetric,
        # DenseEuclideanMetric  # not supported at the moment
    ], τ in [
        StaticTrajectory(lfi, n_steps),
        StaticTrajectory(lfi_jittered, n_steps),
        HMCDA(lf, ϵ * n_steps)
    ]
        n_chains = n_chains_list[i_test]
        metric = metricT((D, n_chains))
        h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        # NoAdaptation
        samples, stats = sample(h, τ, θ_init_list[i_test], n_samples; verbose=false)
        @test mean(samples) ≈ zeros(D, n_chains) atol=RNDATOL * n_chains
        # Adaptation
        for adaptor in [
            Preconditioner(metric),
            NesterovDualAveraging(0.8, lfi.ϵ),
            NaiveHMCAdaptor(
                Preconditioner(metric),
                NesterovDualAveraging(0.8, lfi.ϵ),
            ),
            StanHMCAdaptor(
                Preconditioner(metric),
                NesterovDualAveraging(0.8, lfi.ϵ),
            ),
        ]
            τ isa HMCDA && continue
            samples, stats = sample(h, τ, θ_init_list[i_test], n_samples, adaptor, n_adapts; verbose=false, progress=false)
            @test mean(samples) ≈ zeros(D, n_chains) atol=RNDATOL * n_chains
        end
        # Passing a vector of same RNGs
        rng = [MersenneTwister(1) for _ in 1:n_chains]
        h = Hamiltonian(metricT((D, n_chains)), ℓπ, ∂ℓπ∂θ)
        θ_init = repeat(rand(D), 1, n_chains)
        samples, stats = sample(rng, h, τ, θ_init, n_samples; verbose=false)
        all_same = true
        for i_sample in 2:10
            for j in 2:n_chains
                all_same = all_same && samples[i_sample][:,j] == samples[i_sample][:,1]
            end
        end
        @test all_same
    end
    @info "Adaptation tests for HMCDA with NesterovDualAveraging are skipped"

    # Simple time benchmark
    let metricT=UnitEuclideanMetric
        τ = StaticTrajectory(lf, n_steps)

        time_mat = Vector{Float64}(undef, n_chains_max)
        for (i, n_chains) in enumerate(n_chains_list)
            h = Hamiltonian(metricT((D, n_chains)), ℓπ, ∂ℓπ∂θ)
            t = @elapsed samples, stats = sample(h, τ, θ_init_list[i], n_samples; verbose=false)
            time_mat[i] = t
        end

        # Time for multiple runs of single chain
        time_seperate = Vector{Float64}(undef, n_chains_max)

        for (i, n_chains) in enumerate(n_chains_list)
            t = @elapsed for j in 1:n_chains
                h = Hamiltonian(metricT(D), ℓπ, ∂ℓπ∂θ)
                samples, stats = sample(h, τ, θ_init_list[i][:,j], n_samples; verbose=false)
            end
            time_seperate[i] = t
        end

        # Make plot
        fig = lineplot(
            collect(1:n_chains_max),
            time_mat;
            title="Scalabiliry of multiple chains",
            name="matrix parallism",
            xlabel="Num of chains",
            ylabel="Time (s)"
        )
        lineplot!(fig, collect(n_chains_list), time_seperate; color=:blue, name="seperate")
        println(); show(fig); println(); println()
    end
end
