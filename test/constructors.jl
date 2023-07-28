using AdvancedHMC, AbstractMCMC, Random
include("common.jl")

get_kernel_hyperparams(spl::HMC, state) = [state.κ.τ.L]
get_kernel_hyperparams(spl::HMCDA, state) = [state.κ.τ.λ]
get_kernel_hyperparams(spl::NUTS, state) = [state.κ.τ.max_depth, state.κ.τ.Δ_max]

@testset "Constructors" begin
    d = 2
    θ_init = randn(d)
    model = AbstractMCMC.LogDensityModel(ℓπ_gdemo)

    @testset "$T" for T in [Float32, Float64]
        @testset "$(nameof(typeof(sampler)))" for (sampler, expected) in [
            (
                HMC(T(0.1), 25),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                HMC(25, integrator = Leapfrog(T(0.1))),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                HMC(25, metric = DiagEuclideanMetric(ones(T, 2))),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                HMC(25, integrator = Leapfrog(T(0.1)), metric = :unit),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = UnitEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                HMC(25, integrator = Leapfrog(T(0.1)), metric = :dense),
                (
                    adaptor_type = NoAdaptation,
                    metric_type = DenseEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                HMCDA(T(0.8), one(T), integrator = Leapfrog(T(0.1))),
                (
                    adaptor_type = NesterovDualAveraging,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            # This should perform the correct promotion for the 2nd argument.
            (
                HMCDA(T(0.8), 1, integrator = Leapfrog(T(0.1))),
                (
                    adaptor_type = NesterovDualAveraging,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                NUTS(T(0.8)),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                NUTS(T(0.8); metric = :unit),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = UnitEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                NUTS(T(0.8); metric = :dense),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DenseEuclideanMetric{T},
                    integrator_type = Leapfrog{T},
                ),
            ),
            (
                NUTS(T(0.8); integrator = :jitteredleapfrog),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = JitteredLeapfrog{T,T},
                ),
            ),
            (
                NUTS(T(0.8); integrator = :temperedleapfrog),
                (
                    adaptor_type = StanHMCAdaptor,
                    metric_type = DiagEuclideanMetric{T},
                    integrator_type = TemperedLeapfrog{T,T},
                ),
            ),
        ]
            # Make sure the sampler element type is preserved.
            @test AdvancedHMC.sampler_eltype(sampler) == T

            # Step.
            rng = Random.default_rng()
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
        end
    end

    @testset "Kernel hyperparameters of $(nameof(typeof(sampler)))" for (
        sampler,
        expected,
    ) in [
        (HMC(25), (kernel_hp = [25],)),
        (HMCDA(0.8, 2.0), (kernel_hp = [2.0],)),
        (NUTS(0.8; max_depth = 20, Δ_max = 2000.0), (kernel_hp = [20, 2000.0],)),
    ]

        # Step.
        rng = Random.default_rng()
        transition, state =
            AbstractMCMC.step(rng, model, sampler; n_adapts = 0, init_params = θ_init)

        # Verify that the state is what we expect.

        @test get_kernel_hyperparams(sampler, state) == expected.kernel_hp
    end
end
