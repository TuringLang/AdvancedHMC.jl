module AdvancedHMCCUDAExt

if isdefined(Base, :get_extension)
    import AdvancedHMC
    import CUDA
    import Random
else
    import ..AdvancedHMC
    import ..CUDA
    import ..Random
end

function AdvancedHMC.refresh(
    rng::Union{Random.AbstractRNG,AbstractVector{<:Random.AbstractRNG}},
    ::AdvancedHMC.FullMomentumRefreshment,
    h::AdvancedHMC.Hamiltonian,
    z::AdvancedHMC.PhasePoint{TA},
) where {T<:AbstractFloat,TA<:CUDA.CuArray{<:T}}
    r = CUDA.CuArray{T,2}(undef, size(h.metric)...)
    CUDA.CURAND.randn!(r)
    return AdvancedHMC.phasepoint(h, z.θ, r)
end

# TODO: Ideally this should be merged with the CPU version. The function is 
#       essentially the same but sampling requires a custom call to CUDA.
function AdvancedHMC.mh_accept_ratio(
    rng::Union{Random.AbstractRNG,AbstractVector{<:Random.AbstractRNG}},
    Horiginal::TA,
    Hproposal::TA,
) where {T<:AbstractFloat,TA<:CUDA.CuArray{<:T}}
    α = min.(one(T), exp.(Horiginal .- Hproposal))
    # NOTE: There is a chance that sharing the RNG over multiple
    #       chains for accepting / rejecting might couple
    #       the chains. We need to revisit this more rigirously 
    #       in the future. See discussions at 
    #       https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    r = CUDA.CuArray{T,1}(undef, length(Horiginal))
    CUDA.CURAND.rand!(r)
    accept = r .< α
    return accept, α
end

end # module
