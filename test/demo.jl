using AdvancedHMC, Distributions, ForwardDiff

# Set parameter dimensionality and initial parameter value
dim = 10; θ₀ = rand(dim)

# Define the target distribution
ℓπ(θ) = logpdf(MvNormal(zeros(dim), ones(dim)), θ)

# Set the number of samples to draw and warmup iterations for warmup
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(dim)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
ϵ₀ = find_good_stepsize(hamiltonian, θ₀)
integrator = Leapfrog(ϵ₀)

# Define an HMC sampler, with the following components
#   - (Generalised) no-U-turn criteria for terminating Hamiltonian simulation
#   - Multinomial trajectory sampling scheme
#   - Windowed adaption for (diagonal) mass matrix and step size
kernel = HMCKernel(Trajectory(integrator, GeneralisedNoUTurn()), MultinomialTS)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, ϵ₀))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, kernel, θ₀, n_samples, adaptor, n_adapts; progress=true)
