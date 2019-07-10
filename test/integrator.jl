using Test, AdvancedHMC
include("common.jl")

ϵ = 0.01
lf = Leapfrog(ϵ)

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), ℓπ, ∂ℓπ∂θ)
r_init = AdvancedHMC.rand(h.metric)

n_steps = 10

@testset "step(::Leapfrog) against steps(::Leapfrog)" begin
    z = AdvancedHMC.phasepoint(h, copy(θ_init), copy(r_init))
    z_step = z

    t_step = @elapsed for i = 1:n_steps
        z_step = AdvancedHMC.step(lf, h, z_step)
    end

    t_steps = @elapsed z_steps = AdvancedHMC.step(lf, h, z, n_steps)

    @info "Performance of step() v.s. steps()" n_steps t_step t_steps t_step / t_steps

    @test z_step.θ ≈ z_steps.θ atol=DETATOL
    @test z_step.r ≈ z_steps.r atol=DETATOL
end

# using Turing: Inference
# @testset "steps(::Leapfrog) against Turing.Inference._leapfrog()" begin
#     z = AdvancedHMC.phasepoint(h, θ_init, r_init)
#     t_Turing = @elapsed θ_Turing, r_Turing, _ = Inference._leapfrog(θ_init, r_init, n_steps, ϵ, x -> (nothing, ∂logπ∂θ(x)))
#     t_AHMC = @elapsed z_AHMC = AdvancedHMC.step(lf, h, z, n_steps)
#     @info "Performance of leapfrog of AdvancedHMC v.s. Turing" n_steps t_Turing t_AHMC t_Turing / t_AHMC
#
#     @test θ_Turing ≈ z_AHMC.θ atol=DETATOL
#     @test r_Turing ≈ z_AHMC.r atol=DETATOL
# end

using LinearAlgebra: dot
using Statistics: mean
@testset "Eq (2.11) from (Neal, 2011)" begin
    D = 1
    negU(q::AbstractVector{T}) where {T<:Real} = -dot(q, q) / 2
    ∂negU∂q = q -> (res = GradientResult(q); gradient!(res, negU, q); (value(res), gradient(res)))

    ϵ = 0.01
    lf = Leapfrog(ϵ)

    q_init = randn(D)
    h = Hamiltonian(UnitEuclideanMetric(D), negU, ∂negU∂q)
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
