# Detailed API for AdvancedHMC.jl

An important design goal of AdvancedHMC.jl is modularity; we would like to support algorithmic research on HMC.
This modularity means that different HMC variants can be easily constructed by composing various components, such as preconditioning metric (i.e., mass matrix), leapfrog integrators, trajectories (static or dynamic), adaption schemes, etc. In this documentation, we will explain the detailed usage of different modules in AdancedHMC.jl to provide a comprehensive udnerstanding of how AdvancedHMC.jl can achieve both modularity and efficiency.

### [Hamiltonian mass matrix (`metric`)](@id hamiltonian_mm)

  - Unit metric: `UnitEuclideanMetric(dim)`
  - Diagonal metric: `DiagEuclideanMetric(dim)`
  - Dense metric: `DenseEuclideanMetric(dim)`

where `dim` is the dimensionality of the sampling space.

### [Integrator (`integrator`)](@id integrator)

  - Ordinary leapfrog integrator: `Leapfrog(ϵ)`
  - Jittered leapfrog integrator with jitter rate `n`: `JitteredLeapfrog(ϵ, n)`
  - Tempered leapfrog integrator with tempering rate `a`: `TemperedLeapfrog(ϵ, a)`

where `ϵ` is the step size of leapfrog integration.

### [Kernel (`kernel`)](@id kernel)

  - Static HMC with a fixed number of steps (`n_steps`) from [neal2011mcmc](@Citet): `HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(integrator)))`
  - HMC with a fixed total trajectory length (`trajectory_length`) from [neal2011mcmc](@Citet): `HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(trajectory_length)))`
  - Original NUTS with slice sampling from [hoffman2014no](@Citet): `HMCKernel(Trajectory{SliceTS}(integrator, ClassicNoUTurn()))`
  - Generalised NUTS with slice sampling from [betancourt2017conceptual](@Citet): `HMCKernel(Trajectory{SliceTS}(integrator, GeneralisedNoUTurn()))`
  - Original NUTS with multinomial sampling from [betancourt2017conceptual](@Citet): `HMCKernel(Trajectory{MultinomialTS}(integrator, ClassicNoUTurn()))`
  - Generalised NUTS with multinomial sampling from [betancourt2017conceptual](@Citet): `HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))`

### Adaptor (`adaptor`)

  - Adapt the mass matrix `metric` of the Hamiltonian dynamics: `mma = MassMatrixAdaptor(metric)`
    
      + This is lowered to `UnitMassMatrix`, `WelfordVar` or `WelfordCov` based on the type of the mass matrix `metric`

  - Adapt the step size of the leapfrog integrator `integrator`: `ssa = StepSizeAdaptor(δ, integrator)`
    
      + It uses Nesterov's dual averaging with `δ` as the target acceptance rate.
  - Combine the two above *naively*: `NaiveHMCAdaptor(mma, ssa)`
  - Combine the first two using Stan's windowed adaptation: `StanHMCAdaptor(mma, ssa)`

## The `sample` functions

```julia
sample(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    κ::HMCKernel,
    θ::AbstractVector{<:AbstractFloat},
    n_samples::Int;
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000),
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
)
```

Draw `n_samples` samples using the kernel `κ` under the Hamiltonian system `h`

  - The randomness is controlled by `rng`.
    
      + If `rng` is not provided, the default random number generator (`Random.default_rng()`) will be used.

  - The initial point is given by `θ`.
  - The adaptor is set by `adaptor`, for which the default is no adaptation.
    
      + It will perform `n_adapts` steps of adaptation, for which the default is `1_000` or 10% of `n_samples`, whichever is lower.
  - `drop_warmup` specifies whether to drop samples.
  - `verbose` controls the verbosity.
  - `progress` controls whether to show the progress meter or not.

Note that the function signature of the `sample` function exported by `AdvancedHMC.jl` differs from the [`sample`](https://turinglang.org/dev/docs/using-turing/guide#modelling-syntax-explained) function used by `Turing.jl`. We refer to the documentation of `Turing.jl` for more details on the latter.

Note that the function signature of the `sample` function exported by `AdvancedHMC.jl` differs from the [`sample`](https://turinglang.org/dev/docs/using-turing/guide#modelling-syntax-explained) function used by `Turing.jl`. We refer to the documentation of `Turing.jl` for more details on the latter.

## More types

```@autodocs; canonical=false
Modules = [AdvancedHMC, AdvancedHMC.Adaptation]
Order   = [:type]
```
