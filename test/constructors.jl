using AdvancedHMC, AbstractMCMC, Random
include("common.jl")

@testset "Constructors" begin
    θ_init = randn(2)
    model = AbstractMCMC.LogDensityModel(ℓπ_gdemo)

    @testset "$T" for T in [Float32, Float64]
        @testset "$(nameof(typeof(sampler)))" for sampler in [
            HMC(T(0.1), 25),
            HMCDA(T(0.1), 1),  # this should peform the correct promotion
            NUTS(T(0.8))
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
            @test AdvancedHMC.getmetric(state) isa DiagEuclideanMetric{T, Vector{T}}
            @test AdvancedHMC.getintegrator(state) isa Leapfrog{T}
            # Sampler-specific.
            if sampler isa HMC
                @test AdvancedHMC.getadaptor(state) isa NoAdaptation
            elseif sampler isa HMCDA
                @test AdvancedHMC.getadaptor(state) isa StanHMCAdaptor
            elseif sampler isa NUTS
                @test AdvancedHMC.getadaptor(state) isa StanHMCAdaptor
            else
                error("$(sampler) not recognized")
            end
        end
    end
end
