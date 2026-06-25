using ReTest, AdvancedHMC, LinearAlgebra, Random
using Statistics: mean, var, cov

@testset "sample (vectorized)" begin
    n_chains_max = 20
    n_chains_list = collect(1:n_chains_max)
    θ_init_list = [rand(D, n_chains) for n_chains in n_chains_list]
    ϵ = 0.1
    lf = Leapfrog(ϵ)
    i_test = 5
    lfi = Leapfrog(fill(ϵ, i_test))
    lfi_jittered = JitteredLeapfrog(fill(ϵ, i_test), 1.0)
    n_steps = 10
    n_samples = 20_000
    n_adapts = 4_000

    for metricT in [
            UnitEuclideanMetric,
            DiagEuclideanMetric,
            # DenseEuclideanMetric  # not supported at the moment
        ],
        τ in [
            Trajectory{EndPointTS}(lfi, FixedNSteps(n_steps)),
            Trajectory{MultinomialTS}(lfi, FixedNSteps(n_steps)),
            Trajectory{EndPointTS}(lfi_jittered, FixedNSteps(n_steps)),
            Trajectory{MultinomialTS}(lfi_jittered, FixedNSteps(n_steps)),
            Trajectory{EndPointTS}(lf, FixedIntegrationTime(ϵ * n_steps)),
            Trajectory{MultinomialTS}(lf, FixedIntegrationTime(ϵ * n_steps)),
        ]

        n_chains = n_chains_list[i_test]
        metric = metricT((D, n_chains))
        h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        test_show(metric)
        test_show(h)
        test_show(τ)

        # NoAdaptation
        Random.seed!(100)
        samples, stats = sample(
            h, HMCKernel(τ), θ_init_list[i_test], n_samples; verbose=false
        )
        @test mean(samples) ≈ zeros(D, n_chains) atol = RNDATOL * n_chains

        # Adaptation
        for adaptor in [
            MassMatrixAdaptor(metric),
            StepSizeAdaptor(0.8, lfi),
            NaiveHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, lfi)),
            StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, lfi)),
        ]
            τ.termination_criterion isa FixedIntegrationTime && continue
            test_show(adaptor)

            Random.seed!(100)
            samples, stats = sample(
                h,
                HMCKernel(τ),
                θ_init_list[i_test],
                n_samples,
                adaptor,
                n_adapts;
                verbose=false,
                progress=false,
            )
            @test mean(samples) ≈ zeros(D, n_chains) atol = RNDATOL * n_chains
        end

        # Passing a vector of same RNGs
        rng = [MersenneTwister(1) for _ in 1:n_chains]
        h = Hamiltonian(metricT((D, n_chains)), ℓπ, ∂ℓπ∂θ)
        θ_init = repeat(rand(D), 1, n_chains)
        samples, stats = sample(rng, h, HMCKernel(τ), θ_init, n_samples; verbose=false)
        all_same = true
        for i_sample in 2:10
            for j in 2:n_chains
                all_same = all_same && samples[i_sample][:, j] == samples[i_sample][:, 1]
            end
        end
        @test all_same
    end
    @info "Adaptation tests for FixedIntegrationTime with StepSizeAdaptor are skipped"

    # Simple time benchmark
    let metricT = UnitEuclideanMetric
        κ = HMCKernel(Trajectory{EndPointTS}(lf, FixedNSteps(n_steps)))

        time_mat = Vector{Float64}(undef, n_chains_max)
        for (i, n_chains) in enumerate(n_chains_list)
            h = Hamiltonian(metricT((D, n_chains)), ℓπ, ∂ℓπ∂θ)
            t = @elapsed samples, stats = sample(
                h, κ, θ_init_list[i], n_samples; verbose=false
            )
            time_mat[i] = t
        end

        # Time for multiple runs of single chain
        time_separate = Vector{Float64}(undef, n_chains_max)

        for (i, n_chains) in enumerate(n_chains_list)
            t = @elapsed for j in 1:n_chains
                h = Hamiltonian(metricT(D), ℓπ, ∂ℓπ∂θ)
                samples, stats = sample(
                    h, κ, θ_init_list[i][:, j], n_samples; verbose=false
                )
            end
            time_separate[i] = t
        end

        println("\nVectorized vs separate sampling")
        println("  number of chains:              ", n_chains_list)
        println("  elapsed time [s] (vectorized): ", round.(time_mat; sigdigits=2))
        println("  elapsed time [s] (separate):   ", round.(time_separate; sigdigits=2))
        println(
            "  ratio of elapsed time:         ",
            round.(time_separate ./ time_mat; sigdigits=2),
        )
    end
end
