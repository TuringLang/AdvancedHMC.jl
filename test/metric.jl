using ReTest, Random, AdvancedHMC, LinearAlgebra, Statistics

@testset "Metric" begin
    @testset "Sample momentum variables from metric via vector of RNGs" begin
        D = 10
        n_chains = 5
        θ = randn(D, n_chains)
        rng = [MersenneTwister(1) for _ in 1:n_chains]
        for metric in [
            UnitEuclideanMetric((D, n_chains)),
            DiagEuclideanMetric((D, n_chains)),
            # DenseEuclideanMetric((D, n_chains)) # not supported ATM
            # RankUpdateEuclideanMetric((D, n_chains)) # not supported ATM
        ]
            r = AdvancedHMC.rand_momentum(rng, metric, GaussianKinetic(), θ)
            all_same = true
            for i in 2:n_chains
                all_same = all_same && r[:, i] == r[:, 1]
            end
            @test all_same
        end
    end

    @testset "Resize metric" begin
        D = 10
        rng = MersenneTwister(1)
        θ = randn(rng, D)
        ℓπ(θ) = 1
        for metric in [
            UnitEuclideanMetric(1),
            DiagEuclideanMetric(1),
            DenseEuclideanMetric(1),
            RankUpdateEuclideanMetric(1),
        ]
            h = Hamiltonian(metric, ℓπ, ℓπ)
            h = AdvancedHMC.resize(h, θ)
            @test size(h.metric) == size(θ)
        end
    end

    @testset "RankUpdateEuclideanMetric" begin
        rng = MersenneTwister(1)
        n, k = 5, 2
        A = Diagonal(rand(rng, n) .+ 0.5)
        B = randn(rng, n, k)
        M = randn(rng, k, k)
        Dmat = Symmetric(M * M' + I)
        metric = @inferred RankUpdateEuclideanMetric(A, B, Matrix(Dmat))
        # The full inverse mass matrix `M⁻¹ = A + B D Bᵀ`.
        W = A + B * Dmat * B'
        @test size(metric) == (n,)
        @test eltype(metric) === Float64
        @test AdvancedHMC._diag_inv_metric(metric) ≈ diag(W)

        # Energy and gradient must use the full Woodbury inverse metric, so they have to
        # agree with a `DenseEuclideanMetric` whose inverse mass matrix is `W`.
        dense = DenseEuclideanMetric(Matrix(W))
        ℓπ(θ) = -sum(abs2, θ) / 2
        ∂ℓπ∂θ(θ) = (ℓπ(θ), -θ)
        h_rank = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        h_dense = Hamiltonian(dense, ℓπ, ∂ℓπ∂θ)
        r = randn(rng, n)
        @test (@inferred AdvancedHMC.∂H∂r(h_rank, r)) ≈ AdvancedHMC.∂H∂r(h_dense, r)
        @test (@inferred AdvancedHMC.neg_energy(h_rank, r, r)) ≈
            AdvancedHMC.neg_energy(h_dense, r, r)

        # Sampled momenta have covariance `M = (M⁻¹)⁻¹ = W⁻¹`, which ties `rand_momentum`
        # to the kinetic energy defined above.
        @test (@inferred AdvancedHMC.rand_momentum(rng, metric, GaussianKinetic(), r)) isa
            AbstractVector
        samples = stack(
            AdvancedHMC.rand_momentum(rng, metric, GaussianKinetic(), r) for _ in 1:200_000
        )
        @test cov(samples; dims=2) ≈ inv(W) rtol = 0.05

        @testset "convenience constructors" begin
            # All build the same identity (rank-0) `Float64` metric of size `(n,)`.
            for m in (
                RankUpdateEuclideanMetric(n),
                RankUpdateEuclideanMetric(Float64, n),
                RankUpdateEuclideanMetric((n,)),
                RankUpdateEuclideanMetric(Float64, (n,)),
            )
                @test m isa RankUpdateEuclideanMetric{Float64}
                @test size(m) == (n,)
                @test AdvancedHMC._diag_inv_metric(m) == ones(n)
            end
            m32 = @inferred RankUpdateEuclideanMetric(Float32, n)
            @test eltype(m32) === Float32
            @test AdvancedHMC._diag_inv_metric(m32) == ones(Float32, n)

            # For the rank-0 metric `M⁻¹ = I`, so energy and gradient reduce to the plain
            # Euclidean kinetic term (and must not error on the empty 3-arg `dot`, #1485).
            h0 = Hamiltonian(RankUpdateEuclideanMetric(n), ℓπ, ∂ℓπ∂θ)
            r0 = randn(rng, n)
            @test (@inferred AdvancedHMC.∂H∂r(h0, r0)) ≈ r0
            @test (@inferred AdvancedHMC.neg_energy(h0, r0, r0)) ≈ -sum(abs2, r0) / 2
        end
    end
end
