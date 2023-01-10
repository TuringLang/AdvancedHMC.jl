# Example for Nuts-rs / Nutpie Adaptor
using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra
using Distributions
using Plots
const A = AdvancedHMC

# Define the target distribution using the `LogDensityProblem` interface
struct LogTargetDensity
    dim::Int
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(abs2, θ) / 2  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Choose parameter dimensionality and initial parameter value
D = 20;
initial_θ = rand(D);
ℓπ = LogTargetDensity(D)
n_samples, n_adapts = 2_000, 200

# DEFAULT
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)
proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

@time samples1, stats1 = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true);

# NUTPIE
# https://github.com/pymc-devs/nuts-rs/blob/main/src/adapt_strategy.rs
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)
proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
pc = A.ExpWeightedWelfordVar(size(metric))
adaptor = A.NutpieHMCAdaptor(pc, StepSizeAdaptor(0.8, integrator))
@time samples2, stats2 = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true);

# # Plots
# Plot of variance
get_component(samples, idx) = [samples[i][idx] for i in 1:length(samples)]
# get_component(samples, 3)

# comparison
idx = 10
plot(plot(get_component(samples1, idx), label="Default", color=palette(:default)[1]),
    plot(get_component(samples2, idx), label="Nutpie/Nuts-rs", color=palette(:default)[2]), plot_title=title = "Comparison of component $idx", layout=(2, 1))

# Histogram
pl = histogram(get_component(samples1, idx), label="Default", fillstyle=:\, color=palette(:default)[1], alpha=0.5)
histogram!(pl, get_component(samples2, idx), label="Nutpie/Nuts-rs", alpha=0.5, title="Comparison of component $idx")