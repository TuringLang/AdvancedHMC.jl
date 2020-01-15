using Test, AdvancedHMC, ForwardDiff, Zygote

include("common.jl")

@testset "contrib" begin
    @testset "ad" begin
        metric = UnitEuclideanMetric(D)
        h_hand = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        h_forwarddiff = Hamiltonian(metric, ℓπ, ForwardDiff)
        h_zygote = Hamiltonian(metric, ℓπ, Zygote)
        for x in [rand(D), rand(D, 10)]
            v_hand, g_hand = h_hand.∂ℓπ∂θ(x)
            v_forwarddiff, g_forwarddiff = h_forwarddiff.∂ℓπ∂θ(x)
            v_zygote, g_zygote = h_hand.∂ℓπ∂θ(x)
            @test v_hand ≈ v_forwarddiff
            @test v_hand ≈ v_zygote
            @test g_hand ≈ g_forwarddiff
            @test g_hand ≈ g_forwarddiff
        end
    end
end