using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra
using Distributions

# Define the target distribution using the `LogDensityProblem` interface
struct LogTargetDensity
    dim::Int
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(abs2, θ) / 2  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Choose parameter dimensionality and initial parameter value
D = 10;
initial_θ = rand(D);
ℓπ = LogTargetDensity(D)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)


# Other Example
# Setup
using Distributions: Distributions
using Bijectors: Bijectors
using Random
struct LogDensityDistribution{D<:Distributions.Distribution}
    dist::D
end

LogDensityProblems.dimension(d::LogDensityDistribution) = length(d.dist)
function LogDensityProblems.logdensity(ld::LogDensityDistribution, y)
    d = ld.dist
    b = Bijectors.inverse(Bijectors.bijector(d))
    x, logjac = Bijectors.with_logabsdet_jacobian(b, y)
    return logpdf(d, x) + logjac
end
LogDensityProblems.capabilities(::Type{<:LogDensityDistribution}) = LogDensityProblems.LogDensityOrder{0}()

# Random variance
n_samples, n_adapts = 2_000, 1_000
Random.seed!(1)
D = 10
σ² = 1 .+ abs.(randn(D))

# Diagonal Gaussian
ℓπ = LogDensityDistribution(MvNormal(Diagonal(σ²)))
metric = DiagEuclideanMetric(D)
θ_init = rand(D)
h = Hamiltonian(metric, ℓπ, ForwardDiff)
κ = NUTS(Leapfrog(find_good_stepsize(h, θ_init)))
adaptor = StanHMCAdaptor(
    MassMatrixAdaptor(metric),
    StepSizeAdaptor(0.8, κ.τ.integrator)
)
samples, stats = sample(h, κ, θ_init, n_samples, adaptor, n_adapts; verbose=false)
# adaptor.pc.var ≈ σ²
