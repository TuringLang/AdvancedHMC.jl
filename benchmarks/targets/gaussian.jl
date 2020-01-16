using InteractiveUtils: @code_warntype

function ℓπ_gaussian_impl(m, s, x::AbstractVecOrMat{T}) where {T}
    diff = x .- m
    v = s.^2
    return -(log(2 * T(pi)) .+ log.(v) .+ diff .* diff ./ v) / 2
end

function ℓπ_gaussian(m, s, x::AbstractVector)
    return sum(ℓπ_gaussian_impl(m, s, x))
end

function ℓπ_gaussian(m, s, x::AbstractMatrix)
    return dropdims(sum(ℓπ_gaussian_impl(m, s, x); dims=1); dims=1)
end

function ∂ℓπ∂θ_gaussian_impl(m, s, x::AbstractVecOrMat{T}) where {T}
    diff = x .- m
    v = s.^2
    v = -(log(2 * T(pi)) .+ log.(v) .+ diff .* diff ./ v) / 2
    g = -diff
    return v, g
end

function ∂ℓπ∂θ_gaussian(m, s, x::AbstractVector)
    v, g = ∂ℓπ∂θ_gaussian_impl(m, s, x)
    return sum(v), g
end

function ∂ℓπ∂θ_gaussian(m, s, x::AbstractMatrix)
    v, g = ∂ℓπ∂θ_gaussian_impl(m, s, x)
    return dropdims(sum(v; dims=1); dims=1), g
end

const T = Float32
const gaussian_m = zero(T)
const gaussian_s = one(T)

ℓπ_gaussian(x) = ℓπ_gaussian(gaussian_m, gaussian_s, x)
∂ℓπ∂θ_gaussian(x) = ∂ℓπ∂θ_gaussian(gaussian_m, gaussian_s, x)

function check_typed_gaussian()
    x = zeros(T, 2)
    @code_warntype ℓπ_gaussian(gaussian_m, gaussian_s, x)
    @code_warntype ∂ℓπ∂θ_gaussian(gaussian_m, gaussian_s,x)
    
    x = zeros(T, 2, 2)
    @code_warntype ℓπ_gaussian(gaussian_m, gaussian_s,x)
    @code_warntype ∂ℓπ∂θ_gaussian(gaussian_m, gaussian_s,x)
end

if "--check_typed" in ARGS
    check_typed_gaussian()
end
