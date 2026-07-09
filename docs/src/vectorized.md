# Vectorized HMC Sampling

In this section, we explain how to easily employ vectorized Hamiltonian Monte Carlo with AdvancedHMC.jl. Let's continue with the previous example in [getting-started](@ref get_started), we want to sample a multivariate Gaussian (10-dimensional) with multiple chains, we can simply specify the number of chains in initial parameters, leapfrog integrator, and metric to tell AdvanceHMC.jl how many chains we want to sample. Here, the vectorized multivariate Gaussian log density problem come from [MCMCLogDensityProblems.jl](https://github.com/chalk-lab/MCMCLogDensityProblems.jl) which is a library of common log density target distributions designed for vectorized sampling.

```julia
using AdvancedHMC
using MCMCLogDensityProblems

D = 10
target = HighDimGaussian(D)
ℓπ(x) = logpdf(target, x)
∂ℓπ∂θ(x) = logpdf_grad(target, x)

n_chains = 5
θ_init = rand(D, n_chains)
ϵ = 0.1
lfi = Leapfrog(fill(ϵ, n_chains))
n_steps = 10
n_samples = 20_000
metric = DiagEuclideanMetric((D, n_chains))
τ = Trajectory{EndPointTS}(lfi, FixedNSteps(n_steps))
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
samples, stats = sample(h, HMCKernel(τ), θ_init, n_samples; verbose=false)
```

!!! note
    
    Vectorized sampling only support static HMC, which means samplers like `NUTS` should not be used for vectorized sampling for now.
