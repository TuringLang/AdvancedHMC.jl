import .Zygote

function ∂ℓπ∂θ_zygote(ℓπ, θ::AbstractVector)
    res, back = Zygote.pullback(ℓπ, θ)
    return res, first(back(Zygote.sensitivity(res)))
end

function ∂ℓπ∂θ_zygote(ℓπ, θ::AbstractMatrix)
    res, back = Zygote.pullback(ℓπ, θ)
    return res, first(back(ones(eltype(res), size(res))))
end

function ZygoteADHamiltonian(metric::AbstractMetric, ℓπ)
    ∂ℓπ∂θ(θ::AbstractVecOrMat) = ∂ℓπ∂θ_zygote(ℓπ, θ)
    return Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
end

ADAVAILABLE[Zygote] = ZygoteADHamiltonian
