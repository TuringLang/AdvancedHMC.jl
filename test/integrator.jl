using Test, Random, AdvancedHMC, ForwardDiff
include("common.jl")

ϵ = 0.01
lf = Leapfrog(ϵ)

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), ℓπ, ∂ℓπ∂θ)
r_init = AdvancedHMC.rand(h.metric)

n_steps = 10

@testset "step(::Leapfrog) against step(::Leapfrog)" begin
    z = AdvancedHMC.phasepoint(h, copy(θ_init), copy(r_init))
    z_step = z

    t_step = @elapsed for i = 1:n_steps
        z_step = AdvancedHMC.step(lf, h, z_step)
    end

    t_steps = @elapsed z_steps = AdvancedHMC.step(lf, h, z, n_steps)

    @info "Performance of step() v.s. step()" n_steps t_step t_steps t_step / t_steps

    @test z_step.θ ≈ z_steps.θ atol=DETATOL
    @test z_step.r ≈ z_steps.r atol=DETATOL
end

# using Turing: Inference
# @testset "step(::Leapfrog) against Turing.Inference._leapfrog()" begin
#     z = AdvancedHMC.phasepoint(h, θ_init, r_init)
#     t_Turing = @elapsed θ_Turing, r_Turing, _ = Inference._leapfrog(θ_init, r_init, n_steps, ϵ, x -> (nothing, ∂logπ∂θ(x)))
#     t_AHMC = @elapsed z_AHMC = AdvancedHMC.step(lf, h, z, n_steps)
#     @info "Performance of leapfrog of AdvancedHMC v.s. Turing" n_steps t_Turing t_AHMC t_Turing / t_AHMC
#
#     @test θ_Turing ≈ z_AHMC.θ atol=DETATOL
#     @test r_Turing ≈ z_AHMC.r atol=DETATOL
# end

@testset "jitter" begin
    @testset "Leapfrog" begin
        ϵ0 = 0.1
        lf = Leapfrog(ϵ0)
        @test lf.ϵ == ϵ0
        @test AdvancedHMC.nom_step_size(lf) == ϵ0
        @test AdvancedHMC.step_size(lf) == ϵ0

        lf2 = AdvancedHMC.jitter(Random.GLOBAL_RNG, lf)
        @test lf2 === lf
        @test AdvancedHMC.nom_step_size(lf2) == ϵ0
        @test AdvancedHMC.step_size(lf2) == ϵ0
    end

    @testset "JitteredLeapfrog" begin
        ϵ0 = 0.1
        lf = JitteredLeapfrog(ϵ0, 0.5)
        @test lf.ϵ0 == ϵ0
        @test AdvancedHMC.nom_step_size(lf) == ϵ0
        @test lf.ϵ == ϵ0
        @test AdvancedHMC.step_size(lf) == lf.ϵ

        lf2 = AdvancedHMC.jitter(Random.GLOBAL_RNG, lf)
        @test lf2.ϵ0 == ϵ0
        @test AdvancedHMC.nom_step_size(lf2) == ϵ0
        @test lf2.ϵ != ϵ0
        @test AdvancedHMC.step_size(lf2) == lf2.ϵ
    end
end

@testset "update_nom_step_size" begin
    @testset "Leapfrog" begin
        ϵ0 = 0.1
        lf = Leapfrog(ϵ0)
        @test AdvancedHMC.nom_step_size(lf) == ϵ0

        lf2 = AdvancedHMC.update_nom_step_size(lf, 0.5)
        @test lf2 !== lf
        @test AdvancedHMC.nom_step_size(lf2) == 0.5
        @test AdvancedHMC.step_size(lf2) == 0.5
    end

    @testset "JitteredLeapfrog" begin
        ϵ0 = 0.1
        lf = JitteredLeapfrog(ϵ0, 0.5)
        @test AdvancedHMC.nom_step_size(lf) == ϵ0

        lf2 = AdvancedHMC.update_nom_step_size(lf, 0.2)
        @test lf2 !== lf
        @test AdvancedHMC.nom_step_size(lf2) == 0.2
        # check that we've only updated nominal step size
        @test AdvancedHMC.step_size(lf2) == ϵ0
    end
end

@testset "temper" begin
    αsqrt = 2.0
    lf = TemperedLeapfrog(ϵ, αsqrt ^ 2)
    r = ones(5)
    r1 = AdvancedHMC.temper(lf, r, (i=1, is_half=true), 3)
    r2 = AdvancedHMC.temper(lf, r, (i=1, is_half=false), 3)
    r3 = AdvancedHMC.temper(lf, r, (i=2, is_half=true), 3)
    r4 = AdvancedHMC.temper(lf, r, (i=2, is_half=false), 3)
    r5 = AdvancedHMC.temper(lf, r, (i=3, is_half=true), 3)
    r6 = AdvancedHMC.temper(lf, r, (i=3, is_half=false), 3)
    @test r1 == αsqrt * ones(5)
    @test r2 == αsqrt * ones(5)
    @test r3 == αsqrt * ones(5)
    @test r4 == inv(αsqrt) * ones(5)
    @test r5 == inv(αsqrt) * ones(5)
    @test r6 == inv(αsqrt) * ones(5)
    @test_throws BoundsError AdvancedHMC.temper(lf, r, (i=4, is_half=false), 3)

end

using OrdinaryDiffEq
using LinearAlgebra: dot
using Statistics: mean
@testset "Eq (2.11) from (Neal, 2011)" begin
    D = 1
    negU(q::AbstractVector{T}) where {T<:Real} = -dot(q, q) / 2

    ϵ = 0.01
    for lf in [
        Leapfrog(ϵ),
        DiffEqIntegrator(ϵ, VerletLeapfrog())
    ]
        q_init = randn(D)
        h = Hamiltonian(UnitEuclideanMetric(D), negU, ForwardDiff)
        p_init = AdvancedHMC.rand(h.metric)

        q, p = copy(q_init), copy(p_init)
        z = AdvancedHMC.phasepoint(h, q, p)

        n_steps = 10_000
        qs = zeros(n_steps)
        ps = zeros(n_steps)
        Hs = zeros(n_steps)
        for i = 1:n_steps
            z = AdvancedHMC.step(lf, h, z)
            qs[i] = z.θ[1]
            ps[i] = z.r[1]
            Hs[i] = -AdvancedHMC.neg_energy(z)
        end

        # Throw first 1_000 steps
        qs = qs[1_000:end]
        ps = ps[1_000:end]
        Hs = Hs[1_000:end]

        # Check if all points located at a cirle centered at the origin
        rs = sqrt.(qs.^2 + ps.^2)
        @test all(x-> abs(x - mean(rs)) < 2e-3, rs)

        # Check if the Hamiltonian energy is stable
        @test all(x-> abs(x - mean(Hs)) < 2e-3, Hs)
    end
end
