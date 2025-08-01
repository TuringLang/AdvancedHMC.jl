const AbstractScalarOrVec{T} = Union{T,AbstractVector{T}} where {T<:AbstractFloat}

# Support of passing a vector of RNGs

function _randn(rng::AbstractRNG, ::Type{T}, n_chains::Int) where {T}
    return randn(rng, T, n_chains)
end
function _randn(rng::AbstractRNG, ::Type{T}, dim::Int, n_chains::Int) where {T}
    return randn(rng, T, dim, n_chains)
end

function _randn(rngs::AbstractVector{<:AbstractRNG}, ::Type{T}, n_chains::Int) where {T}
    @argcheck length(rngs) == n_chains
    return map(Base.Fix2(randn, T), rngs)
end
function _randn(
    rngs::AbstractVector{<:AbstractRNG}, ::Type{T}, dim::Int, n_chains::Int
) where {T}
    @argcheck length(rngs) == n_chains
    out = similar(rngs, T, dim, n_chains)
    foreach(Random.randn!, rngs, eachcol(out))
    return out
end

"""
    __axes(r::AbstractVecOrMat)

Return the axes of input `r`, where `r` can be `AbstractArrays`, `ComponentArrays` or other custom arrays.
"""
@inline __axes(r::AbstractVecOrMat) = axes(r)

""" 
`rand_coupled` produces coupled randomness given a vector of RNGs. For example, 
when a vector of RNGs is provided, `rand_coupled` peforms a single `rand` call 
(rather than a `rand` call for each RNG) while keep all RNGs synchronised. 
This is important if we want to couple multiple Markov chains.
"""

rand_coupled(rng::AbstractRNG, args...) = rand(rng, args...)

function rand_coupled(rngs::AbstractVector{<:AbstractRNG}, args...)
    # Dummpy calles to sync RNGs
    foreach(rngs[2:end]) do rng
        rand(rng, args...)
    end
    return res = rand(first(rngs), args...)
end

# Sample from Categorical distributions

function randcat(rng::AbstractRNG, p::AbstractVector{T}) where {T}
    u = rand(rng, T)
    c = zero(eltype(p))
    i = 0
    while c < u
        c += p[i += 1]
    end
    return max(i, 1)
end

"""
    randcat(rng, P::AbstractMatrix)

Generating Categorical random variables in a vectorized mode.
`P` is supposed to be a matrix of (D, N) where each column is a probability vector.

Example

```
P = [
    0.5 0.3;
    0.4 0.6;
    0.1 0.1
]
u = [0.3, 0.4]
C = [
    0.5 0.3
    0.9 0.9
    1.0 1.0
]
```
Then `C .< u'` is
```
[
    0 1
    0 0
    0 0
]
```
thus `convert.(Int, vec(sum(C .< u'; dims=1))) .+ 1` equals `[1, 2]`.
"""
function randcat(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}}, P::AbstractMatrix{T}
) where {T}
    u = if rng isa AbstractRNG
        rand(rng, T, size(P, 2))
    else
        @argcheck length(rng) == size(P, 2)
        map(Base.Fix2(rand, T), rng)
    end
    C = cumsum(P; dims=1)
    return max.(vec(count(C .< u'; dims=1)) .+ 1, 1)  # prevent numerical issue for Float32
end
