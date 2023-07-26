using AdvancedHMC, AbstractMCMC, Random
include("common.jl")

@testset "Constructors" begin
    θ_init = randn(2)
    model = AbstractMCMC.LogDensityModel(ℓπ_gdemo)

    @testset "$T" for T in [Float32, Float64]
        @testset "$(nameof(typeof(sampler)))" for (sampler, expected) in [
            (
                HMC(T(0.1), 25),
                (
                    adaptor_type=NoAdaptation,
                    metric_type=DiagEuclideanMetric{T},
                    integrator_type=Leapfrog{T},
                )
            ),
            # This should peform the correct promotion for the 2nd argument.
            (
                HMCDA(T(0.1), 1),
                (
                    adaptor_type=StanHMCAdaptor,
                    metric_type=DiagEuclideanMetric{T},
                    integrator_type=Leapfrog{T},
                )
            ),
            (
                NUTS(T(0.8)),
                (
                    adaptor_type=StanHMCAdaptor,
                    metric_type=DiagEuclideanMetric{T},
                    integrator_type=Leapfrog{T},
                )
            ),
            (
                NUTS(T(0.8); metric = :unit),
                (
                    adaptor_type=StanHMCAdaptor,
                    metric_type=UnitEuclideanMetric{T},
                    integrator_type=Leapfrog{T},
                )
            ),
            (
                NUTS(T(0.8); metric = :dense),
                (
                    adaptor_type=StanHMCAdaptor,
                    metric_type=DenseEuclideanMetric{T},
                    integrator_type=Leapfrog{T},
                )
            ),
        ]
            # Make sure the sampler element type is preserved.
            @test AdvancedHMC.sampler_eltype(sampler) == T

            # Step.
            rng = Random.default_rng()
            transition, state = AbstractMCMC.step(
                rng, model, sampler; n_adapts = 0, init_params = θ_init
            )

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
end
