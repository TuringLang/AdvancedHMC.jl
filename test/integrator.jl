using Test, HMC
include("common.jl")

h = Hamiltonian(UnitMetric(), _logπ, _dlogπdθ)
ϵ = 0.01
lf = Leapfrog(ϵ)

θ_init = randn(D)
r_init = HMC.rand_momentum(h, θ_init)

n_steps = 10

@testset "step() against steps()" begin
    θ1, r1 = copy(θ_init), copy(r_init)

    for i = 1:n_steps
        θ1, r1, _ = HMC.step(lf, h, θ1, r1)
    end

    θ2, r2, _ = HMC.steps(lf, h, θ_init, r_init, n_steps)

    @test θ1 ≈ θ2 atol=DETATOL
    @test r1 ≈ r2 atol=DETATOL
end

using Turing: Inference
@testset "steps() against Turing.Inference._leapfrog()" begin
    θ1, r1, _ = Inference._leapfrog(θ_init, r_init, n_steps, ϵ, x -> (nothing, -_dlogπdθ(x)))
    θ2, r2, _ = HMC.steps(lf, h, θ_init, r_init, n_steps)

    @test θ1 ≈ θ2 atol=DETATOL
    @test r1 ≈ r2 atol=DETATOL
end
