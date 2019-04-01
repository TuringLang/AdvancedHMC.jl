using Test, AdvancedHMC
include("common.jl")

ϵ = 0.01
lf = Leapfrog(ϵ)

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), logπ, ∂logπ∂θ)
r_init = AdvancedHMC.rand_momentum(h)

n_steps = 10

@testset "step(::Leapfrog) against steps(::Leapfrog)" begin
    θ_step, r_step = copy(θ_init), copy(r_init)

    t_step = @elapsed for i = 1:n_steps
        θ_step, r_step, _ = AdvancedHMC.step(lf, h, θ_step, r_step)
    end

    t_steps = @elapsed θ_steps, r_steps, _ = AdvancedHMC.steps(lf, h, θ_init, r_init, n_steps)

    @info "Performance of step() v.s. steps()" n_steps t_step t_steps t_step / t_steps

    @test θ_step ≈ θ_steps atol=DETATOL
    @test r_step ≈ r_steps atol=DETATOL
end

using Turing: Inference
@testset "steps(::Leapfrog) against Turing.Inference._leapfrog()" begin
    t_Turing = @elapsed θ_Turing, r_Turing, _ = Inference._leapfrog(θ_init, r_init, n_steps, ϵ, x -> (nothing, ∂logπ∂θ(x)))
    t_AHMC = @elapsed θ_AHMC, r_AHMC, _ = AdvancedHMC.steps(lf, h, θ_init, r_init, n_steps)
    @info "Performance of leapfrog of AdvancedHMC v.s. Turing" n_steps t_Turing t_AHMC t_Turing / t_AHMC

    @test θ_Turing ≈ θ_AHMC atol=DETATOL
    @test r_Turing ≈ r_AHMC atol=DETATOL
end

using LinearAlgebra: dot
using Statistics: mean
@testset "Eq (2.11) from Neal (2011)" begin
    D = 1
    negU(q::AbstractVector{T}) where {T<:Real} = -dot(q, q) / 2
    ∂negU∂q = q -> gradient(negU, q)

    ϵ = 0.01
    lf = Leapfrog(ϵ)

    q_init = randn(D)
    h = Hamiltonian(UnitEuclideanMetric(D), negU, ∂negU∂q)
    p_init = AdvancedHMC.rand_momentum(h)

    q, p = copy(q_init), copy(p_init)

    n_steps = 10_000
    qs = zeros(n_steps)
    ps = zeros(n_steps)
    Hs = zeros(n_steps)
    for i = 1:n_steps
        q, p, _ = AdvancedHMC.step(lf, h, q, p)
        qs[i] = q[1]
        ps[i] = p[1]
        Hs[i] = AdvancedHMC.hamiltonian_energy(h, q, p)
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
