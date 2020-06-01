const ADSUPPORT = (:ForwardDiff, :Zygote)
const ADAVAILABLE = Dict{Module, Function}()

Hamiltonian(metric::M, ℓπ::T, m::Module) where {M<:AbstractMetric,T} = ADAVAILABLE[m](metric, ℓπ)

function Hamiltonian(metric::AbstractMetric, ℓπ)
    available = collect(keys(ADAVAILABLE))
    if length(available) == 0
        support_list_str = join(ADSUPPORT, " or ")
        error("MethodError: no method matching Hamiltonian(metric::AbstractMetric, ℓπ) because no backend is loaded. Please load an AD package ($support_list_str) first.")
    elseif length(available) > 1
        available_list_str = join(keys(ADAVAILABLE), " and ")
        constructor_list_str = join(map(m -> "Hamiltonian(metric, ℓπ, $m)", available), "\n  ")
        error("MethodError: Hamiltonian(metric::AbstractMetric, ℓπ) is ambiguous because multiple AD pakcages are available ($available_list_str). Please use AD explictly. Candidates:\n  $constructor_list_str")
    else
        return Hamiltonian(metric, ℓπ, first(available))
    end
end
