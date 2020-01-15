### Hand-coded multivaite Gaussain

const D = 5
const gaussian_m = zeros(D)
const gaussian_s = ones(D)

function ℓπ(m, s, x::AbstractVecOrMat{T}) where {T}
    diff = x .- m
    v = s.^2
    return -(log(2 * T(pi)) .+ log.(v) .+ diff .* diff ./ v) / 2
end

function ℓπ(θ::AbstractVector)
    return sum(ℓπ(gaussian_m, gaussian_s, θ))
end

function ℓπ(θ::AbstractMatrix)
    return dropdims(sum(ℓπ(gaussian_m, gaussian_s, θ); dims=1); dims=1)
end

function ∂ℓπ∂θ(m, s, x::AbstractVecOrMat{T}) where {T}
    diff = x .- m
    v = s.^2
    v = -(log(2 * T(pi)) .+ log.(v) .+ diff .* diff ./ v) / 2
    g = -diff
    return v, g
end

function ∂ℓπ∂θ(θ::AbstractVector)
    v, g = ∂ℓπ∂θ(gaussian_m, gaussian_s, θ)
    return sum(v), g
end

function ∂ℓπ∂θ(θ::AbstractMatrix)
    v, g = ∂ℓπ∂θ(gaussian_m, gaussian_s, θ)
    return dropdims(sum(v; dims=1); dims=1), g
end

### Testing parameters

# Tolerance ratio
const TRATIO = Int == Int64 ? 1 : 2
# Deterministic tolerance
const DETATOL = 1e-3 * D * TRATIO
# Random tolerance
const RNDATOL = 5e-2 * D * TRATIO

### NUTS helper

using ForwardDiff
using Random: GLOBAL_RNG

function run_nuts(dim::Int, ℓπ::Function; rng=GLOBAL_RNG, ∂ℓπ∂θ=ForwardDiff, metric=DiagEuclideanMetric(dim), n_samples=5_000, n_adapts=2_000, verbose=false, drop_warmup=false)
    initial_θ = randn(rng, dim)

    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    
    integrator = Leapfrog(0.1)

    proposal = NUTS(integrator)
    adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, integrator))

    samples, stats = sample(rng, hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; verbose=verbose, drop_warmup=drop_warmup)

    return (samples=samples, stats=stats, adaptor=adaptor)
end
