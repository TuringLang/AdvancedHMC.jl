using Test, AdvancedHMC, LinearAlgebra, UnicodePlots
using Statistics: mean, var, cov
include("common.jl")

@testset "Matrix parallelisation" begin
    n_chains_max = 20
    θ_init = [randn(D, n_chains) for n_chains in 1:n_chains_max]
    ϵ = 0.1
    lf = Leapfrog(ϵ)
    n_steps = 20
    n_samples = 10_000
    n_adapts = 2_000

    for metricT in [
        UnitEuclideanMetric,
        DiagEuclideanMetric,
        # DenseEuclideanMetric  # not supported at the moment
    ], τ in [
        StaticTrajectory(lf, n_steps),
        HMCDA(lf, ϵ * n_steps)
    ]
        metric = metricT((D, 5))
        h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        samples, stats = sample(h, τ, θ_init[5], n_samples; verbose=false)
        @test mean(samples) ≈ zeros(D, 5) atol=RNDATOL

        for adaptor in [
            Preconditioner(metric),
            # NesterovDualAveraging(0.8, lf.ϵ),
            # NaiveHMCAdaptor(
            #     Preconditioner(metric),
            #     NesterovDualAveraging(0.8, lf.ϵ),
            # ),
            # StanHMCAdaptor(
            #     n_adapts,
            #     Preconditioner(metric),
            #     NesterovDualAveraging(0.8, lf.ϵ),
            # ),
        ]
            samples, stats = sample(h, τ, θ_init[5], n_samples, adaptor, n_adapts; verbose=false, progress=false)
            @test mean(samples) ≈ zeros(D, 5) atol=RNDATOL
        end
    end

    # Simple time benchmark
    let metricT=UnitEuclideanMetric
        τ = StaticTrajectory(lf, n_steps)

        time_mat = Vector{Float64}(undef, n_chains_max)
        for (i, n_chains) in enumerate(1:n_chains_max)
            h = Hamiltonian(metricT((D, n_chains)), ℓπ, ∂ℓπ∂θ)
            t = @elapsed samples, stats = sample(h, τ, θ_init[i], n_samples; verbose=false)
            time_mat[i] = t
        end

        # Time for multiple runs of single chain
        time_seperate = Vector{Float64}(undef, n_chains_max)

        for (i, n_chains) in enumerate(1:n_chains_max)
            t = @elapsed for j in 1:n_chains
                h = Hamiltonian(metricT(D), ℓπ, ∂ℓπ∂θ)
                samples, stats = sample(h, τ, θ_init[i][:,j], n_samples; verbose=false)
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
        lineplot!(fig, collect(1:n_chains_max), time_seperate; color=:blue, name="seperate")
        println(); show(fig); println(); println()
    end
end
