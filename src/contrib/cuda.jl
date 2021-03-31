import .CUDA

CUDA.allowscalar(false)

function refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::FullMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint{T}
) where {T<:CUDA.CuArray}
    r = CUDA.CuArray{Float32, 2}(undef, size(h.metric)...)
    CUDA.CURAND.randn!(r)
    return phasepoint(h, z.θ, r)
end

function mh_accept_ratio(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, Horiginal::TA, Hproposal::TA,
) where {T<:AbstractFloat, TA<:CUDA.CuArray{<:T}}
    α = min.(one(T), exp.(Horiginal .- Hproposal))
    # NOTE: There is a chance that sharing the RNG over multiple
    #       chains for accepting / rejecting might couple
    #       the chains. We need to revisit this more rigirously 
    #       in the future. See discussions at 
    #       https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    accept = TA(rand(rng, T, length(Horiginal))) .< α
    return accept, α
end