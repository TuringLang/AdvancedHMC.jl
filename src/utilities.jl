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

# Helper functions to sync RNGs that ensures coupling works properly and reproducible

rand_sync(rng::AbstractRNG, args...) = rand(rng, args...)

# This function is needed to use only one RNG while a vector of RNGs are provided
# in a way that RNGs are still synchronised. This is important if we want to use 
# same RNGs to couple multiple chains.
function rand_sync(rngs::AbstractVector{<:AbstractRNG}, args...)
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
        c += p[i+=1]
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
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, 
    P::AbstractMatrix{T}
) where {T}
    u = rand(rng, T, size(P, 2))
    C = cumsum(P; dims=1)
    is = convert.(Int, vec(sum(C .< u'; dims=1))) .+ 1
    return max.(is, 1)  # prevent numerical issue for Float32
end
