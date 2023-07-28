using AdvancedHMC, AbstractMCMC, Random
include("common.jl")

get_kernel_hyperparams(spl::HMC, state) = state.κ.τ.termination_criterion.L
get_kernel_hyperparams(spl::HMCDA, state) = state.κ.τ.termination_criterion.λ
get_kernel_hyperparams(spl::NUTS, state) =
    state.κ.τ.termination_criterion.max_depth, state.κ.τ.termination_criterion.Δ_max

get_kernel_hyperparamsT(spl::HMC, state) = typeof(state.κ.τ.termination_criterion.L)
get_kernel_hyperparamsT(spl::HMCDA, state) = typeof(state.κ.τ.termination_criterion.λ)
get_kernel_hyperparamsT(spl::NUTS, state) = typeof(state.κ.τ.termination_criterion.Δ_max)

@testset "Constructors" begin
    d = 2
    θ_init = randn(d)
    rng = Random.default_rng()
    model = AbstractMCMC.LogDensityModel(ℓπ_gdemo)

    @testset "$T" for T in [Float32, Float64]
        @testset "$(nameof(typeof(sampler)))" for (sampler, expected) in [
            (
                HMC(T(0.1), 25),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = 25,
                ),
            ),
            (
                HMC(25, integrator = Leapfrog(T(0.1))),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = 25,
                ),
            ),
            (
                HMC(25, metric = DiagEuclideanMetric(ones(T, 2))),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = 25,
                ),
            ),
            (
                HMC(25, integrator = Leapfrog(T(0.1)), metric = :unit),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = UnitEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = 25,
                ),
            ),
            (
                HMC(25, integrator = Leapfrog(T(0.1)), metric = :dense),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DenseEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = 25,
                ),
            ),
            (
                HMCDA(T(0.8), one(T), integrator = Leapfrog(T(0.1))),
                (
                    adaptor_type = NesterovDualAveraging,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = one(T),
                ),
            ),
            # This should perform the correct promotion for the 2nd argument.
            (
                HMCDA(T(0.8), 1, integrator = Leapfrog(T(0.1))),
                (
                    adaptor_type = NesterovDualAveraging,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = one(T),
                ),
            ),
            (
                NUTS(T(0.8); max_depth = 20, Δ_max = T(2000.0)),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = (20, T(2000.0)),
                ),
            ),
            (
                NUTS(T(0.8); metric = :unit),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = UnitEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = (10, T(1000.0)),
                ),
            ),
            (
                NUTS(T(0.8); metric = :dense),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DenseEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                    kernel_hp = (10, T(1000.0)),
                ),
            ),
            (
                NUTS(T(0.8); integrator = :jitteredleapfrog),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = JitteredLeapfrog{T,T},
                    kernel_hp = (10, T(1000.0)),
                ),
            ),
            (
                NUTS(T(0.8); integrator = :temperedleapfrog),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = TemperedLeapfrog{T,T},
                    kernel_hp = (10, T(1000.0)),
                ),
            ),
        ]
            # Make sure the sampler element type is preserved.
            @test AdvancedHMC.sampler_eltype(sampler) == T

            # Step.
            transition, state =
                AbstractMCMC.step(rng, model, sampler; n_adapts = 0, init_params = θ_init)

            # Verify that the types are preserved in the transition.
            @test eltype(transition.z.θ) == T
            @test eltype(transition.z.r) == T
            @test eltype(transition.z.ℓπ.value) == T
            @test eltype(transition.z.ℓπ.gradient) == T
            @test eltype(transition.z.ℓκ.value) == T
            @test eltype(transition.z.ℓκ.gradient) == T

            # Verify that the state is what we expect.
            @test AdvancedHMC.getmetric(state) isa expected.metric_type
            @test AdvancedHMC.getintegrator(state) isa expected.integrator_type
            @test AdvancedHMC.getadaptor(state) isa expected.adaptor_type

            # Verify that the kernel is receiving the hyperparameters
            @test get_kernel_hyperparams(sampler, state) == expected.kernel_hp
            if typeof(sampler) <: HMC
                @test get_kernel_hyperparamsT(sampler, state) == Int64
            else
                @test get_kernel_hyperparamsT(sampler, state) == T
            end
        end
    end
end

@testset "Utils" begin
    @testset "init_params" begin
        d = 2
        θ_init = randn(d)
        rng = Random.default_rng()
        model = AbstractMCMC.LogDensityModel(ℓπ_gdemo)
        logdensity = model.logdensity
        spl = NUTS(0.8)
        T = sampler_eltype(spl)

        metric = make_metric(spl, logdensity)
        hamiltonian = Hamiltonian(metric, model)

        init_params1 = make_init_params(rng, spl, logdensity, nothing)
        @test typeof(init_params1) == Vector{T}
        @test length(init_params1) == d
        init_params2 = make_init_params(rng, spl, logdensity, θ_init)
        @test init_params2 === θ_init
    end
end
