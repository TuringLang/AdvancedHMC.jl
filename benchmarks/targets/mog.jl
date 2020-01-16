using InteractiveUtils: @code_warntype

include("gaussian.jl")

function density_mog_impl(m, s, mixing, x::AbstractArray)
    return dropdims(sum(exp.(ℓπ_gaussian(mog_m, mog_s, x)) .* mixing; dims=1); dims=1)
end

density_mog(m, s, mixing, x::AbstractVector) = density_mog_impl(m, s, mixing, x)

function density_mog(m, s, mixing, x::AbstractMatrix)
    dim, n = size(x)
    x = reshape(x, dim, 1, n)
    return density_mog_impl(m, s, ximing, x)
end

ℓπ_mog(m, s, mixing, x) = log.(density_mog(m, s, mixing, x))

const T = Float32
const mog_m  = T[-2.0, 0.0, 3.2, 2.5]
const mog_s  = T[ 1.2, 1.0, 5.0, 2.8]
const mixing = T[ 0.2, 0.3, 0.1, 0.4]

ℓπ_mog(x) = ℓπ_mog(mog_m, mog_s, mixing, x)

function check_typed_mog()
    x = zeros(T, 2)
    @code_warntype ℓπ_mog(x)
    
    x = zeros(T, 2, 2)
    @code_warntype ℓπ_mog(x)
end

if "--check_typed" in ARGS
    check_typed_mog()
end
