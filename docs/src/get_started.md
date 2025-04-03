# Sampling from a multivariate Gaussian using NUTS

In this section, we demonstrate a minimal example of sampling from a multivariate Gaussian (10-dimensional) using the No U-Turn Sampler (NUTS). Below we describe the major components of the Hamiltonian system which are essential to sample using this approach:

  - **Metric**: In many sampling problems the sample space is associated with a metric that allows us to measure the distance between any two points, and other similar quantities. In the example in this section, we use a special metric called the **Euclidean Metric**, represented with a `D × D` matrix from which we can compute distances.[^1]

  - **Leapfrog integration**: Leapfrog integration is a second-order numerical method for integrating differential equations (In this case they are equations of motion for the relative position of one particle with respect to the other). The order of this integration signifies its rate of convergence. Any algorithm with a finite time step size will have numerical errors, and the order is related to this error. For a second-order algorithm, this error scales as the second power of the time step, hence the name. High-order integrators are usually complex to code and have a limited region of convergence; thus they do not allow arbitrarily large time steps. A second-order integrator is suitable for our purpose. Hence we opt for the leapfrog integrator. It is called `leapfrog` due to the ways this algorithm is written, where the positions and velocities of particles "leap over" each other.[^2]
  - **Kernel for trajectories (static or dynamic)**: Different kernels, which may be static or dynamic, can be used. At each iteration of any variant of the HMC algorithm, there are two main steps - the first step changes the momentum and the second step may change both the position and the momentum of a particle.[^3]

```julia
using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra

# Define the target distribution using the `LogDensityProblem` interface
struct LogTargetDensity
    dim::Int
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(abs2, θ) / 2  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
function LogDensityProblems.capabilities(::Type{LogTargetDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

# Choose parameter dimensionality and initial parameter value
D = 10;
initial_θ = rand(D);
ℓπ = LogTargetDensity(D)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with the initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(
    hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true
)
```

## Parallel sampling

AdvancedHMC enables parallel sampling (either distributed or multi-thread) via Julia's [parallel computing functions](https://docs.julialang.org/en/v1/manual/parallel-computing/).
It also supports vectorized sampling for static HMC.

The below example utilizes the `@threads` macro to sample 4 chains across 4 threads.

```julia
# Ensure that Julia was launched with an appropriate number of threads
println(Threads.nthreads())

# Number of chains to sample
nchains = 4

# Cache to store the chains
chains = Vector{Any}(undef, nchains)

# The `samples` from each parallel chain is stored in the `chains` vector 
# Adjust the `verbose` flag as per need
Threads.@threads for i in 1:nchains
    samples, stats = sample(
        hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; verbose=false
    )
    chains[i] = samples
end
```

## Using the `AbstractMCMC` interface

Users can also use the `AbstractMCMC` interface to sample, which is also used in Turing.jl.
In order to show how this is done let us start from our previous example where we defined a `LogTargetDensity`, `ℓπ`.

```julia
using AbstractMCMC, LogDensityProblemsAD
# Wrap the previous LogTargetDensity as LogDensityModel 
# where ℓπ::LogTargetDensity
model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ))

# Wrap the previous sampler as a HMCSampler <: AbstractMCMC.AbstractSampler
D = 10;
initial_θ = rand(D);
n_samples, n_adapts, δ = 1_000, 2_000, 0.8
sampler = HMCSampler(kernel, metric, adaptor)

# Now sample
samples = AbstractMCMC.sample(
    model, sampler, n_adapts + n_samples; n_adapts=n_adapts, initial_params=initial_θ
)
```

## Convenience Constructors

In the previous examples, we built the sampler by manually specifying the integrator, metric, kernel, and adaptor to build our own sampler. However, in many cases, users might want to initialize a standard NUTS sampler. In such cases having to define each of these aspects manually is tedious and error-prone. For these reasons `AdvancedHMC` also provides users with a series of convenience constructors for standard samplers. We will now show how to use them.

  - HMC:
    
    ```julia
    # HMC Sampler
    # step size, number of leapfrog steps 
    n_leapfrog, ϵ = 25, 0.1
    hmc = HMC(ϵ, n_leapfrog)
    ```
    
    Equivalent to:
    
    ```julia
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    integrator = Leapfrog(0.1)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_leapfrog)))
    adaptor = NoAdaptation()
    hmc = HMCSampler(kernel, metric, adaptor)
    ```

  - NUTS:
    
    ```julia
    # NUTS Sampler
    # adaptation steps, target acceptance probability,
    δ = 0.8
    nuts = NUTS(δ)
    ```
    
    Equivalent to:
    
    ```julia
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(δ, integrator))
    nuts = HMCSampler(kernel, metric, adaptor)
    ```
  - HMCDA:
    
    ```julia
    #HMCDA (dual averaging)
    # adaptation steps, target acceptance probability, target trajectory length 
    δ, λ = 0.8, 1.0
    hmcda = HMCDA(δ, λ)
    ```
    
    Equivalent to:
    
    ```julia
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(λ)))
    adaptor = StepSizeAdaptor(δ, initial_ϵ)
    hmcda = HMCSampler(kernel, metric, adaptor)
    ```

Moreover, there's some flexibility in how these samplers can be initialized.
For example, a user can initialize a NUTS (HMC and HMCDA) sampler with their own metrics and integrators.
This can be done as follows:

```julia
nuts = NUTS(δ; metric=:diagonal) #metric = DiagEuclideanMetric(D) (Default!)
nuts = NUTS(δ; metric=:unit)     #metric = UnitEuclideanMetric(D)
nuts = NUTS(δ; metric=:dense)    #metric = DenseEuclideanMetric(D)
# Provide your own AbstractMetric
metric = DiagEuclideanMetric(10)
nuts = NUTS(δ; metric=metric)

nuts = NUTS(δ; integrator=:leapfrog)         #integrator = Leapfrog(ϵ) (Default!)
nuts = NUTS(δ; integrator=:jitteredleapfrog) #integrator = JitteredLeapfrog(ϵ, 0.1ϵ)
nuts = NUTS(δ; integrator=:temperedleapfrog) #integrator = TemperedLeapfrog(ϵ, 1.0)

# Provide your own AbstractIntegrator
integrator = JitteredLeapfrog(0.1, 0.2)
nuts = NUTS(δ; integrator=integrator)
```

## GPU Sampling with CUDA

There is experimental support for running static HMC on the GPU using CUDA.
To do so, the user needs to have [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installed, ensure the logdensity of the `Hamiltonian` can be executed on the GPU and that the initial points are a `CuArray`.
A small working example can be found at `test/cuda.jl`.

## Footnotes

[^1]: The Euclidean metric is also known as the mass matrix in the physical perspective. See [Hamiltonian mass matrix](@ref hamiltonian-mm) for available metrics.
[^2]: About the leapfrog integration scheme: Suppose ${\bf x}$ and ${\bf v}$ are the position and velocity of an individual particle respectively; $i$ and $i+1$ are the indices for time values $t_i$ and $t_{i+1}$ respectively; $dt = t_{i+1} - t_i$ is the time step size (constant and regularly spaced intervals), and ${\bf a}$ is the acceleration induced on a particle by the forces of all other particles. Furthermore, suppose positions are defined at times $t_i, t_{i+1}, t_{i+2}, \dots $, spaced at constant intervals $dt$, the velocities are defined at halfway times in between, denoted by $t_{i-1/2}, t_{i+1/2}, t_{i+3/2}, \dots $, where $t_{i+1} - t_{i + 1/2} = t_{i + 1/2} - t_i = dt / 2$, and the accelerations ${\bf a}$ are defined only on integer times, just like the positions. Then the leapfrog integration scheme is given as: $x_{i} = x_{i-1} + v_{i-1/2} dt; \quad v_{i+1/2} = v_{i-1/2} + a_i dt$. For available integrators refer to [Integrator](@ref integrator).
[^3]: On kernels: In the classical HMC approach, during the first step, new values for the momentum variables are randomly drawn from their Gaussian distribution, independently of the current values of the position variables. A Metropolis update is performed during the second step, using Hamiltonian dynamics to provide a new state. For available kernels refer to [Kernel](@ref kernel).
