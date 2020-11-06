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
    indices = convert.(Int, vec(sum(C .< u'; dims=1))) .+ 1
    return max.(indices, 1)  # prevent numerical issue for Float32
end

"""
    or!(x, y)

Compute elementwise OR between `x` and `y` (i.e. `x .| y`), updating `x` in-place if
possible. Note that if `x` is an `AbstractArray`, then `x .| y` must have the same size as
`x`.
"""
or!(x, y) = x .| y
or!(x::AbstractArray, y) = x .|= y

"""
    eitheror!(a, b, tf)

Given a number or array `a`, a number or array `b`, and a boolean or array of booleans `tf`,
return a number or container containing values of `a` wherever `tf` is true and values of
`b` wherever `tf` is false.

If `a` is an `AbstractArray`, then the result is stored in `a`, and the dimensions of `a`,
`b`, and `tf` must be compatible with the output being stored in `a`.
"""
@inline eitheror!(a, b, tf) = ifelse.(tf, a, b)
@inline function eitheror!(a::AbstractArray, b, tf)
    a .= ifelse.(tf, a, b)
    return a
end

@inline accept!(x, x′, is_accept::Bool) = ifelse(is_accept, x′, x)
@inline accept!(x, x′, is_accept) = eitheror!(x′, x, is_accept)
@inline function accept!(x::AbstractMatrix, x′::AbstractMatrix, is_accept::AbstractVector)
    return eitheror!(x′, x, is_accept')
end

@inline colwise_dot(x::AbstractVector, y::AbstractVector) = dot(x, y)
# TODO: this function needs a custom GPU kernel, e.g. using KernelAbstractions
function colwise_dot(x::AbstractMatrix, y::AbstractMatrix)
    T = Base.promote_eltypeof(x, y)
    z = similar(x, T, size(x, 2))
    @inbounds @simd for i in eachindex(z)
        z[i] = dot(@view(x[:, i]), @view(y[:, i]))
    end
    return z
end
