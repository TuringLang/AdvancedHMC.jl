# AdvancedHMC.jl

[![CI](https://github.com/TuringLang/AdvancedHMC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/TuringLang/AdvancedHMC.jl/actions/workflows/CI.yml)
[![DOI](https://zenodo.org/badge/72657907.svg)](https://zenodo.org/badge/latestdoi/72657907)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AdvancedHMC.jl/badge.svg?branch=kx%2Fbug-fix)](https://coveralls.io/github/TuringLang/AdvancedHMC.jl?branch=kx%2Fbug-fix)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/AdvancedHMC.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/AdvancedHMC.jl/dev/)

AdvancedHMC.jl provides a robust, modular, and efficient implementation of advanced HMC algorithms. An illustrative example of AdvancedHMC's usage is given below. AdvancedHMC.jl is part of [Turing.jl](https://github.com/TuringLang/Turing.jl), a probabilistic programming library in Julia. 
If you are interested in using AdvancedHMC.jl through a probabilistic programming language, please check it out!

**Interfaces**
- [`IMP.hmc`](https://github.com/salilab/hmc): an experimental Python module for the Integrative Modeling Platform, which uses AdvancedHMC in its backend to sample protein structures.

**NEWS**
- We presented a paper for AdvancedHMC.jl at [AABI](http://approximateinference.org/) in 2019 in Vancouver, Canada. ([abs](http://proceedings.mlr.press/v118/xu20a.html), [pdf](http://proceedings.mlr.press/v118/xu20a/xu20a.pdf), [OpenReview](https://openreview.net/forum?id=rJgzckn4tH))
- We presented a poster for AdvancedHMC.jl at [StanCon 2019](https://mc-stan.org/events/stancon2019Cambridge/) in Cambridge, UK. ([pdf](https://github.com/TuringLang/AdvancedHMC.jl/files/3730367/StanCon-AHMC.pdf))

**API CHANGES**
- [v0.5.0] **Breaking!** Convenience constructors for common samplers changed to:
  - `HMC(leapfrog_stepsize::Real, n_leapfrog::Int)`
  - `NUTS(target_acceptance::Real)` 
  - `HMCDA(target_acceptance::Real, integration_time::Real)`
- [v0.2.22] Three functions are renamed.
  - `Preconditioner(metric::AbstractMetric)` -> `MassMatrixAdaptor(metric)` and 
  - `NesterovDualAveraging(δ, integrator::AbstractIntegrator)` -> `StepSizeAdaptor(δ, integrator)`
  - `find_good_eps` -> `find_good_stepsize`
- [v0.2.15] `n_adapts` is no longer needed to construct `StanHMCAdaptor`; the old constructor is deprecated.
- [v0.2.8] Two Hamiltonian trajectory sampling methods are renamed to avoid a name clash with Distributions.
  - `Multinomial` -> `MultinomialTS`
  - `Slice` -> `SliceTS`
- [v0.2.0] The gradient function passed to `Hamiltonian` is supposed to return a value-gradient tuple now.

## A minimal example - sampling from a multivariate Gaussian using NUTS

This section demonstrates a minimal example of sampling from a multivariate Gaussian (10-dimensional) using the no U-turn sampler (NUTS). Below we describe the major components of the Hamiltonian system which are essential to sample using this approach:

- **Metric**: In many sampling problems the sample space is associated with a metric that allows us to measure the distance between any two points, and other similar quantities. In the example in this section, we use a special metric called the **Euclidean Metric**, represented with a `D × D` matrix from which we can compute distances.[^1]

- **Leapfrog integration**: Leapfrog integration is a second-order numerical method for integrating differential equations (In this case they are equations of motion for the relative position of one particle with respect to the other). The order of this integration signifies its rate of convergence. Any algorithm with a finite time step size will have numerical errors, and the order is related to this error. For a second-order algorithm, this error scales as the second power of the time step, hence, the name second-order. High-order integrators are usually complex to code and have a limited region of convergence; hence they do not allow arbitrarily large time steps. A second-order integrator is suitable for our purpose. Hence we opt for the leapfrog integrator. It is called `leapfrog` due to the ways this algorithm is written, where the positions and velocities of particles "leap over" each other.[^2]

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
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)
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
samples, stats = sample(hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true)
```

### Parallel sampling 

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
  samples, stats = sample(hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; verbose=false)
  chains[i] = samples
end
```

### Using the `AbstractMCMC` interface 

Users can also use the `AbstractMCMC` interface to sample, which is also used in Turing.jl.
In order to show how this is done let us start from our previous example where we defined a `LogTargetDensity`, `ℓπ`.

```julia
using AbstractMCMC, LogDensityProblemsAD
# Wrap the previous LogTargetDensity as LogDensityModel 
# where ℓπ::LogTargetDensity
model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ))

# Wrap the previous sampler as a HMCSampler <: AbstractMCMC.AbstractSampler
D = 10; initial_θ = rand(D)
n_samples, n_adapts, δ = 1_000, 2_000, 0.8
sampler = HMCSampler(kernel, metric, adaptor) 

# Now sample
samples = AbstractMCMC.sample(
      model,
      sampler,
      n_adapts + n_samples;
      n_adapts = n_adapts,
      initial_params = initial_θ,
  )
```

### Convenience Constructors

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
  kernel =  HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
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
nuts = NUTS(δ, metric = :diagonal) #metric = DiagEuclideanMetric(D) (Default!)
nuts = NUTS(δ, metric = :unit)     #metric = UnitEuclideanMetric(D)
nuts = NUTS(δ, metric = :dense)    #metric = DenseEuclideanMetric(D)
# Provide your own AbstractMetric
metric = DiagEuclideanMetric(10)
nuts = NUTS(δ, metric = metric)

nuts = NUTS(δ, integrator = :leapfrog)         #integrator = Leapfrog(ϵ) (Default!)
nuts = NUTS(δ, integrator = :jitteredleapfrog) #integrator = JitteredLeapfrog(ϵ, 0.1ϵ)
nuts = NUTS(δ, integrator = :temperedleapfrog) #integrator = TemperedLeapfrog(ϵ, 1.0)

# Provide your own AbstractIntegrator
integrator = JitteredLeapfrog(0.1, 0.2)
nuts = NUTS(δ, integrator = integrator) 
```

### GPU Sampling with CUDA

There is experimental support for running static HMC on the GPU using CUDA. 
To do so, the user needs to have [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installed, ensure the logdensity of the `Hamiltonian` can be executed on the GPU and that the initial points are a `CuArray`. 
A small working example can be found at `test/cuda.jl`.

## API and supported HMC algorithms

An important design goal of AdvancedHMC.jl is modularity; we would like to support algorithmic research on HMC.
This modularity means that different HMC variants can be easily constructed by composing various components, such as preconditioning metric (i.e., mass matrix), leapfrog integrators,  trajectories (static or dynamic), adaption schemes, etc. 
The minimal example above can be modified to suit particular inference problems by picking components from the list below.

### Hamiltonian mass matrix (`metric`)

- Unit metric: `UnitEuclideanMetric(dim)`
- Diagonal metric: `DiagEuclideanMetric(dim)`
- Dense metric: `DenseEuclideanMetric(dim)`

where `dim` is the dimensionality of the sampling space.

### Integrator (`integrator`)

- Ordinary leapfrog integrator: `Leapfrog(ϵ)`
- Jittered leapfrog integrator with jitter rate `n`: `JitteredLeapfrog(ϵ, n)`
- Tempered leapfrog integrator with tempering rate `a`: `TemperedLeapfrog(ϵ, a)`

where `ϵ` is the step size of leapfrog integration.

### Kernel (`kernel`)

- Static HMC with a fixed number of steps (`n_steps`) (Neal, R. M. (2011)): `HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(integrator)))`
- HMC with a fixed total trajectory length (`trajectory_length`) (Neal, R. M. (2011)): `HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(trajectory_length)))` 
- Original NUTS with slice sampling (Hoffman, M. D., & Gelman, A. (2014)): `HMCKernel(Trajectory{SliceTS}(integrator, ClassicNoUTurn()))`
- Generalised NUTS with slice sampling (Betancourt, M. (2017)): `HMCKernel(Trajectory{SliceTS}(integrator, GeneralisedNoUTurn()))`
- Original NUTS with multinomial sampling (Betancourt, M. (2017)): `HMCKernel(Trajectory{MultinomialTS}(integrator, ClassicNoUTurn()))`
- Generalised NUTS with multinomial sampling (Betancourt, M. (2017)): `HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))`

### Adaptor (`adaptor`)

- Adapt the mass matrix `metric` of the Hamiltonian dynamics: `mma = MassMatrixAdaptor(metric)`
  - This is lowered to `UnitMassMatrix`, `WelfordVar` or `WelfordCov` based on the type of the mass matrix `metric`
- Adapt the step size of the leapfrog integrator `integrator`: `ssa = StepSizeAdaptor(δ, integrator)`
  - It uses Nesterov's dual averaging with `δ` as the target acceptance rate.
- Combine the two above *naively*: `NaiveHMCAdaptor(mma, ssa)`
- Combine the first two using Stan's windowed adaptation: `StanHMCAdaptor(mma, ssa)`

### Gradients 
`AdvancedHMC` supports AD-based using [`LogDensityProblemsAD`](https://github.com/tpapp/LogDensityProblemsAD.jl) and user-specified gradients. In order to use user-specified gradients, please replace `ForwardDiff` with `ℓπ_grad` in the `Hamiltonian`  constructor, where the gradient function `ℓπ_grad` should return a tuple containing both the log-posterior and its gradient. 

All the combinations are tested in [this file](https://github.com/TuringLang/AdvancedHMC.jl/blob/master/test/sampler.jl) except for using tempered leapfrog integrator together with adaptation, which we found unstable empirically.

## The `sample` function signature in detail

```julia
function sample(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    κ::HMCKernel,
    θ::AbstractVector{<:AbstractFloat},
    n_samples::Int,
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
)
```

Draw `n_samples` samples using the kernel `κ` under the Hamiltonian system `h`

- The randomness is controlled by `rng`.
  - If `rng` is not provided, `GLOBAL_RNG` will be used.
- The initial point is given by `θ`.
- The adaptor is set by `adaptor`, for which the default is no adaptation.
  - It will perform `n_adapts` steps of adaptation, for which the default is `1_000` or 10% of `n_samples`, whichever is lower. 
- `drop_warmup` specifies whether to drop samples.
- `verbose` controls the verbosity.
- `progress` controls whether to show the progress meter or not.

Note that the function signature of the `sample` function exported by `AdvancedHMC.jl` differs from the [`sample`](https://turinglang.org/dev/docs/using-turing/guide#modelling-syntax-explained) function used by `Turing.jl`. We refer to the documentation of `Turing.jl` for more details on the latter.

## Citing AdvancedHMC.jl ##
If you use AdvancedHMC.jl for your own research, please consider citing the following publication:

Kai Xu, Hong Ge, Will Tebbutt, Mohamed Tarek, Martin Trapp, Zoubin Ghahramani: "AdvancedHMC.jl: A robust, modular and efficient implementation of advanced HMC algorithms.", *Symposium on Advances in Approximate Bayesian Inference*, 2020. ([abs](http://proceedings.mlr.press/v118/xu20a.html), [pdf](http://proceedings.mlr.press/v118/xu20a/xu20a.pdf))

with the following BibTeX entry:

```
@inproceedings{xu2020advancedhmc,
  title={AdvancedHMC. jl: A robust, modular and efficient implementation of advanced HMC algorithms},
  author={Xu, Kai and Ge, Hong and Tebbutt, Will and Tarek, Mohamed and Trapp, Martin and Ghahramani, Zoubin},
  booktitle={Symposium on Advances in Approximate Bayesian Inference},
  pages={1--10},
  year={2020},
  organization={PMLR}
}
```

If you using AdvancedHMC.jl directly through Turing.jl, please consider citing the following publication:

Hong Ge, Kai Xu, and Zoubin Ghahramani: "Turing: a language for flexible probabilistic inference.", *International Conference on Artificial Intelligence and Statistics*, 2018. ([abs](http://proceedings.mlr.press/v84/ge18b.html), [pdf](http://proceedings.mlr.press/v84/ge18b/ge18b.pdf))

with the following BibTeX entry:

```
@inproceedings{ge2018turing,
  title={Turing: A language for flexible probabilistic inference},
  author={Ge, Hong and Xu, Kai and Ghahramani, Zoubin},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1682--1690},
  year={2018},
  organization={PMLR}
}
```

## References

1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))

2. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

3. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2), 123-214. ([arXiv](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x))

4. Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

5. Betancourt, M. (2016). Identifying the optimal integration time in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1601.00225](https://arxiv.org/abs/1601.00225).

6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([arXiv](http://arxiv.org/abs/1111.4246))

## Footnotes

[^1]: The Euclidean metric is also known as the mass matrix in the physical perspective. See [Hamiltonian mass matrix](#Hamiltonian-mass-matrix-(metric)) for available metrics.
[^2]: About the leapfrog integration scheme: Suppose ${\bf x}$ and ${\bf v}$ are the position and velocity of an individual particle respectively; $i$ and $i+1$ are the indices for time values $t_i$ and $t_{i+1}$ respectively; $dt = t_{i+1} - t_i$ is the time step size (constant and regularly spaced intervals), and ${\bf a}$ is the acceleration induced on a particle by the forces of all other particles. Furthermore, suppose positions are defined at times $t_i, t_{i+1}, t_{i+2}, \dots $, spaced at constant intervals $dt$, the velocities are defined at halfway times in between, denoted by $t_{i-1/2}, t_{i+1/2}, t_{i+3/2}, \dots $, where $t_{i+1} - t_{i + 1/2} = t_{i + 1/2} - t_i = dt / 2$, and the accelerations ${\bf a}$ are defined only on integer times, just like the positions. Then the leapfrog integration scheme is given as: $x_{i} = x_{i-1} + v_{i-1/2} dt; \quad v_{i+1/2} = v_{i-1/2} + a_i dt$. For available integrators refer to [Integrator](#Integrator-(integrator)).
[^3]: On kernels: In the classical HMC approach, during the first step, new values for the momentum variables are randomly drawn from their Gaussian distribution, independently of the current values of the position variables. A Metropolis update is performed during the second step, using Hamiltonian dynamics to provide a new state. For available kernels refer to [Kernel](#Kernel-(kernel)).
