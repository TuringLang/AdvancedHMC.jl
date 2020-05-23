const AbstractScalarOrVec{T} = Union{T,AbstractVector{T}} where {T<:AbstractFloat}

# Support of passing a vector of RNGs

Base.rand(rng::AbstractVector{<:AbstractRNG}) = rand.(rng)

Base.randn(rng::AbstractVector{<:AbstractRNG}) = randn.(rng)

function Base.rand(rng::AbstractVector{<:AbstractRNG}, T, n_chains::Int)
    @argcheck length(rng) == n_chains
    return rand.(rng, T)
end

function Base.randn(rng::AbstractVector{<:AbstractRNG}, T, dim::Int, n_chains::Int)
    @argcheck length(rng) == n_chains
    return cat(randn.(rng, T, dim)...; dims=2)
end

# Sample from Categorical distributions

function randcat(rng::AbstractRNG, p::AbstractVector{T}) where {T}
    u = rand(rng, T)
    c = zero(eltype(p))
    i = 0
    while c < u
        c += p[i+=1]
    end
    return max(i, 1)
end

function randcat(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, 
    P::AbstractMatrix{T}
) where {T}
    u = rand(rng, T, size(P, 1))
    C = cumsum(P; dims=2)
    is = convert.(Int, vec(sum(C .< u; dims=2))) .+ 1
    return max.(is, 1)
end
