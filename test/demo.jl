### Define the target distribution
using Distributions: logpdf, MvNormal

D = 10
target = MvNormal(zeros(D), ones(D))
ℓπ(θ) = logpdf(target, θ)

### Build up a HMC sampler to draw samples
using AdvancedHMC, ForwardDiff  # or, using Zygote
                                # AdvancedHMC will use loaded AD package for gradient
# Sampling parameter settings
n_samples, n_adapts = 12_000, 2_000

# Draw a random starting points
θ_init = rand(D)

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ) # or, Hamiltonian(metric, ℓπ, ∂ℓπ∂θ) for hand-coded gradient ∂ℓπ∂θ
integrator = Leapfrog(find_good_eps(h, θ_init))
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, int))

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples, stats = sample(h, prop, θ_init, n_samples, adaptor, n_adapts; progress=true)