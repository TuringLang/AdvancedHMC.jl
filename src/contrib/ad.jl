const ADSUPPORT = (:ForwardDiff, :Zygote, :ReverseDiff)
const ADAVAILABLE = Dict{Module, Function}()

Hamiltonian(metric::AbstractMetric, ℓπ, m::Module) = ADAVAILABLE[m](metric, ℓπ)

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

### ForwardDiff

@require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin

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
#     logdensities = similar(θ, size(θ, 2))   # pre-allocate logdensities; this is important
#     function ℓπ_sum(x)
#         logdensities .= ℓπ(x)
#         return sum(logdensities)
#     end
#     res = DiffResults.GradientResult(θ)
#     ForwardDiff.gradient!(res, ℓπ_sum, θ)
#     return ForwardDiff.value.(logdensities), DiffResults.gradient(res)
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

end # @require

### Zygote

@require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin

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

end # @require

### ReverseDiff

@require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin

import .ReverseDiff, .ReverseDiff.DiffResults

function get_∂ℓπ∂θ_reversediff(ℓπ, θ::AbstractVector)
    inputs = (θ,)
    f_tape = ReverseDiff.GradientTape(ℓπ, inputs)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    results = similar.(inputs)
    all_results = DiffResults.GradientResult.(results)
    # cfg = ReverseDiff.GradientConfig(inputs)    # we may use this in the future; see https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl#L43
    function ∂ℓπ∂θ(θ::AbstractVector)
        ReverseDiff.gradient!(all_results, compiled_f_tape, (θ,))
        return DiffResults.value(first(all_results)), DiffResults.gradient(first(all_results))
    end
    return ∂ℓπ∂θ
end

# Implementation 1
function get_∂ℓπ∂θ_reversediff(ℓπ, θ::AbstractMatrix)
    logdensities = similar(θ, size(θ, 2))   # pre-allocate logdensities; this is important
    function ℓπ_sum(x)
        logdensities .= ℓπ(x)
        return sum(logdensities)
    end
    inputs = (θ,)
    f_tape = ReverseDiff.GradientTape(ℓπ_sum, inputs)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    results = similar.(inputs)
    all_results = DiffResults.GradientResult.(results)
    # cfg = ReverseDiff.GradientConfig(inputs)    # we may use this in the future; see https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl#L43
    function ∂ℓπ∂θ(θ::AbstractMatrix)
        ReverseDiff.gradient!(all_results, compiled_f_tape, (θ,))
        return ReverseDiff.value(logdensities), DiffResults.gradient(first(all_results))
    end
    return ∂ℓπ∂θ
end

# Implementation 2 (not finished as it's too slow already without the transformation, compared to Impl. 1)
# function get_∂ℓπ∂θ_reversediff(ℓπ, θ::AbstractMatrix)
#     inputs = (θ,)
#     f_tape = ReverseDiff.JacobianTape(ℓπ, inputs)
#     compiled_f_tape = ReverseDiff.compile(f_tape)
#     results = similar.(inputs, size(θ, 2))
#     all_results = DiffResults.JacobianResult.(results, inputs)
#     # cfg = ReverseDiff.JacobianConfig(inputs)    # we may use this in the future; see https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl#L43
#     function ∂ℓπ∂θ(θ::AbstractMatrix)
#         ReverseDiff.jacobian!(all_results, compiled_f_tape, (θ,))
#         return DiffResults.value(first(all_results)), DiffResults.jacobian(first(all_results))
#     # FIXME: transform the Jacob to vectorized form and return
#     end
#     return ∂ℓπ∂θ
# end

function ReverseDiffADHamiltonian(metric::AbstractMetric, ℓπ)
    θ = rand(metric)
    ∂ℓπ∂θ = get_∂ℓπ∂θ_reversediff(ℓπ, θ)
    return Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
end

ADAVAILABLE[ReverseDiff] = ReverseDiffADHamiltonian

end # @require
