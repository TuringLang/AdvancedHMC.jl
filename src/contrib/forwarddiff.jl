import .ForwardDiff, .ForwardDiff.DiffResults

function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractVector)
    res = DiffResults.GradientResult(θ)
    ForwardDiff.gradient!(res, ℓπ, θ)
    return DiffResults.value(res), DiffResults.gradient(res)
end

# Implementation 1
function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractMatrix)
    jacob = similar(θ)
    res = DiffResults.JacobianResult(similar(θ, size(θ, 2)), jacob)
    ForwardDiff.jacobian!(res, ℓπ, θ)
    jacob_full = DiffResults.jacobian(res)

    d, n = size(jacob)
    for i in 1:n
        jacob[:,i] = jacob_full[i,1+(i-1)*d:i*d]
    end
    return DiffResults.value(res), jacob
end

# Implementation 2
# function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractMatrix)
#     local densities
#     f(x) = (densities = ℓπ(x); sum(densities))
#     res = DiffResults.GradientResult(θ)
#     ForwardDiff.gradient!(res, f, θ)
#     return ForwardDiff.value.(densities), DiffResults.gradient(res)
# end

# Implementation 3
# function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractMatrix)
#     v = similar(θ, size(θ, 2))
#     g = similar(θ)
#     for i in 1:size(θ, 2)
#         res = GradientResult(θ[:,i])
#         gradient!(res, ℓπ, θ[:,i])
#         v[i] = value(res)
#         g[:,i] = gradient(res)
#     end
#     return v, g
# end

function ForwardDiffHamiltonian(metric::AbstractMetric, ℓπ)
    ∂ℓπ∂θ(θ::AbstractVecOrMat) = ∂ℓπ∂θ_forwarddiff(ℓπ, θ)
    return Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
end

ADAVAILABLE[ForwardDiff] = ForwardDiffHamiltonian
