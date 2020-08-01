using AdvancedHMC, Distributions, ForwardDiff

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)

# Define the target distribution
ℓπ(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)

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
trajectory = Trajectory(integrator, FixedNSteps(10))
trajectory = Trajectory(integrator, NoUTurn())
kernel = HMCKernel(FullRefreshment(), trajectory, MultinomialTS)

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, kernel, initial_θ, n_samples; progress=true)
