module AdvancedHMCADTypesExt

using AdvancedHMC: AbstractMetric, Hamiltonian
using AbstractMCMC: LogDensityModel
using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD
using ADTypes: AbstractADType

function Hamiltonian(
    metric::AbstractMetric, ℓπ::LogDensityModel, kind::AbstractADType; kwargs...
)
    return Hamiltonian(metric, ℓπ.logdensity, kind; kwargs...)
end
function Hamiltonian(metric::AbstractMetric, ℓπ, kind::AbstractADType; kwargs...)
    if LogDensityProblems.capabilities(ℓπ) === nothing
        throw(
            ArgumentError(
                "The log density function does not support the LogDensityProblems.jl interface",
            ),
        )
    end
    ℓ = LogDensityProblemsAD.ADgradient(kind, ℓπ; kwargs...)
    return Hamiltonian(metric, ℓ)
end

end
