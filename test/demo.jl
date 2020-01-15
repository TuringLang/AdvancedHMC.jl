### Define the target distribution
using Distributions: logpdf, MvNormal

D = 10
ℓπ(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)

### Build up a HMC sampler to draw samples
using AdvancedHMC, ForwardDiff

# Parameter settings
n_samples, n_adapts = 12_000, 2_000
θ₀ = rand(D)    # draw a random starting points

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)  # or, Hamiltonian(metric, ℓπ, ∂ℓπ∂θ) for hand-coded gradient ∂ℓπ∂θ
integrator = Leapfrog(find_good_eps(hamiltonian, θ₀))
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, integrator))

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples, stats = sample(hamiltonian, proposal, θ₀, n_samples, adaptor, n_adapts; progress=true)