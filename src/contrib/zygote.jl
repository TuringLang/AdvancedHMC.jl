import .Zygote

function ∇ℓπ_zygote(ℓπ, θ::AbstractVector)
    res, back = Zygote.pullback(ℓπ, θ)
    return res, first(back(Zygote.sensitivity(res)))
end

function ∇ℓπ_zygote(ℓπ, θ::AbstractMatrix)
    res, back = Zygote.pullback(ℓπ, θ)
    return res, first(back(ones(eltype(res), size(res))))
end

function ZygoteADHamiltonian(metric::AbstractMetric, ℓπ)
    ∇ℓπ(θ::AbstractVecOrMat) = ∇ℓπ_zygote(ℓπ, θ)
    return Hamiltonian(metric, ℓπ, ∇ℓπ)
end

ADAVAILABLE[Zygote] = ZygoteADHamiltonian
