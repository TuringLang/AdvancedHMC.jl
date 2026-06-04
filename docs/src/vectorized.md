# [Vectorized HMC Sampling](@id vectorized_hmc)

When you want to run many Markov chains over the *same* target distribution, AdvancedHMC.jl lets you sample all of them at once in a single, vectorized run. Rather than looping over the chains one at a time, their states are stacked together into a batched parameter array, conventionally of shape `D × n_chains`, where each column holds the current position of one chain. A single call to `sample` then advances every chain together using batched array operations.

In this section we explain what vectorized sampling is, how it differs from running chains in parallel, when it is a good fit, and a few limitations worth keeping in mind. We will build on the [getting-started example](@ref get_started) and again sample from a 10-dimensional multivariate Gaussian, but this time with several chains at once.

## Vectorized versus parallel sampling

It is worth being clear about the difference between *vectorized* sampling and *parallel* sampling, since both let you run more than one chain.

Parallel sampling, described under [Parallel Sampling](@ref parallel_sampling) in the getting-started guide, launches a separate, independent run of the sampler for each chain and spreads those runs across threads or processes. Every chain executes the ordinary scalar sampler, and the speedup comes from carrying out several of those runs at the same time on different workers.

Vectorized sampling instead stays within a single process and a single sampler call. The speedup comes from expressing each chain's update as one operation over a batched array, which can take advantage of SIMD, BLAS, broadcasting, or GPU execution. Because all chains share the same code path, they advance in lockstep: every chain takes a leapfrog step together, refreshes its momentum together, and so on.

As a rule of thumb, reach for parallel sampling when each chain needs its own independent control flow, and reach for vectorized sampling when the chains can all march in step and you want to exploit fast batched-array hardware.

## What your target needs to provide

To run in vectorized mode, each ingredient simply needs to carry the chain dimension:

  - `θ_init` should be a `D × n_chains` matrix, with one column per chain.
  - The step size can be a single scalar shared by all chains, or a per-chain vector such as `fill(ϵ, n_chains)`.
  - The metric should include the chain dimension, for example `DiagEuclideanMetric((D, n_chains))`.
  - The log density function should accept the `D × n_chains` matrix and return one log-density value per chain (a length-`n_chains` vector).
  - The gradient function supplied to `Hamiltonian` should return a tuple `(logdensity_values, gradient)`, where `logdensity_values` is a length-`n_chains` vector and `gradient` is a `D × n_chains` matrix.

Here, the vectorized multivariate Gaussian target comes from [MCMCLogDensityProblems.jl](https://github.com/chalk-lab/MCMCLogDensityProblems.jl), a library of common log-density target distributions designed for vectorized sampling.

```julia
using AdvancedHMC
using MCMCLogDensityProblems

D = 10
target = HighDimGaussian(D)
ℓπ(x) = logpdf(target, x)
# logpdf_grad returns (logpdf_values, gradient_matrix), as required by AdvancedHMC
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

## When to use it

Vectorized sampling tends to shine when

  - you want many chains of the same target, all with the same dimension;
  - the target is naturally written to accept batched matrix inputs;
  - you would like to run on a GPU or otherwise lean on batched-array performance; and
  - you want to avoid the overhead of launching and managing many separate sampler runs.

If your chains have different dimensions, need chain-specific control flow, or your log-density function can only handle one parameter vector at a time, vectorized sampling is not the right tool, and parallel sampling is a better fit.

## A note on randomness

Unless you pass a random number generator explicitly, sampling uses Julia's default RNG. In vectorized mode each chain draws its own variates from that single stream rather than being fed identical noise. A few internal operations do deliberately share randomness across chains, though: the trajectory direction in multinomial trajectories is currently coupled, and the accept/reject step draws from one shared stream. Vectorized sampling is therefore best understood as batched multi-chain sampling rather than a guarantee of completely independent RNG streams in every internal operation. If you want more explicit per-chain control or reproducibility, you can pass a vector of RNGs, one per chain, which `sample` accepts directly.

!!! note
    
    Vectorized sampling currently supports static HMC only. Dynamic samplers such as `NUTS` build their trajectories adaptively, using per-chain doubling and U-turn checks, so different chains would do different amounts of work and could not be advanced as a single lockstep batch. Dense metrics are likewise not yet supported in vectorized (matrix) mode.
