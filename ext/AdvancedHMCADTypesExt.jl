module AdvancedHMCADTypesExt

using AdvancedHMC: AdvancedHMC, AbstractMetric, Hamiltonian, LogDensityModel
using AdvancedHMC: LogDensityProblems, LogDensityProblemsAD
using ADTypes: AbstractADType

function AdvancedHMC.Hamiltonian(
    metric::AbstractMetric, ℓπ::LogDensityModel, kind::AbstractADType; kwargs...
)
    return Hamiltonian(metric, ℓπ.logdensity, kind; kwargs...)
end
function AdvancedHMC.Hamiltonian(
    metric::AbstractMetric, ℓπ, kind::AbstractADType; kwargs...
)
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
