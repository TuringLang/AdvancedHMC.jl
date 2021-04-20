import .CUDA

CUDA.allowscalar(false)

function refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::FullMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint{TA}
) where {T<:AbstractFloat,TA<:CUDA.CuArray{<:T}}
    r = CUDA.CuArray{T, 2}(undef, size(h.metric)...)
    CUDA.CURAND.randn!(r)
    return phasepoint(h, z.θ, r)
end

# TODO: Ideally this should be merged with the CPU version. The function is 
#       essentially the same but sampling requires a custom call to CUDA.
function mh_accept_ratio(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, Horiginal::TA, Hproposal::TA,
) where {T<:AbstractFloat, TA<:CUDA.CuArray{<:T}}
    α = min.(one(T), exp.(Horiginal .- Hproposal))
    # NOTE: There is a chance that sharing the RNG over multiple
    #       chains for accepting / rejecting might couple
    #       the chains. We need to revisit this more rigirously 
    #       in the future. See discussions at 
    #       https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    r = CUDA.CuArray{T, 1}(undef, length(Horiginal))
    CUDA.CURAND.rand!(r)
    accept = r .< α
    return accept, α
end