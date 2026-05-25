# Detailed API for AdvancedHMC.jl

An important design goal of AdvancedHMC.jl is modularity; we would like to support algorithmic research on HMC.
This modularity means that different HMC variants can be easily constructed by composing various components, such as preconditioning metric (i.e., mass matrix), leapfrog integrators, trajectories (static or dynamic), adaption schemes, etc. In this section, we will explain the detailed usage of different modules in AdancedHMC.jl to provide a comprehensive udnerstanding of how AdvancedHMC.jl can achieve both modularity and efficiency. The section highlights the key components of AdvancedHMC.jl, with a complete documentation provided at the end.

### [Hamiltonian mass matrix (`metric`)](@id hamiltonian_mm)

  - Unit metric: `UnitEuclideanMetric(dim)`
  - Diagonal metric: `DiagEuclideanMetric(dim)`
  - Dense metric: `DenseEuclideanMetric(dim)`

where `dim` is the dimension of the sampling space.

Two experimental position-dependent (Riemannian) metrics are also available:

  - `RiemannianMetric((dim,), calc_G, calc_∂G∂θ)` — for user-supplied positive-definite
    metrics `G(θ)` (e.g. Fisher information). `calc_G` should return either a plain
    `Matrix` or an `AbstractPDMat` (preferred — reuses the stored Cholesky). `calc_∂G∂θ`
    returns the `(d, d, d)` tensor `∂G/∂θ`.
  - `SoftAbsRiemannianMetric((dim,), calc_H, calc_∂H∂θ, α)` — for Hessian-based metrics
    where `H(θ)` is not guaranteed to be positive definite. The SoftAbs transformation
    `G = Q · diag(λ · coth(αλ)) · Qᵀ` (Betancourt, 2012) regularises `H`'s eigenvalues
    to a strictly positive spectrum. `α` controls how closely SoftAbs approximates `|λ|`.

The legacy `DenseRiemannianMetric(dim, G, ∂G∂θ[, map])` constructor is deprecated and
forwards to the appropriate type above based on whether `map` is `IdentityMap()` or
`SoftAbsMap(α)`.

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
      + There is an experimental way to improve the *diagonal* mass matrix adaptation using gradient information (similar to [nutpie](https://github.com/pymc-devs/nutpie)),
        currently to be initialized for a `metric` of type `DiagEuclideanMetric`
        via `mma = AdvancedHMC.NutpieVar(size(metric); var=copy(metric.M⁻¹))`
        until a new interface is introduced in an upcoming breaking release to specify the method of adaptation.

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

## Full documentation of APIs in AdvancedHMC.jl

```@autodocs; canonical=false
Modules = [AdvancedHMC, AdvancedHMC.Adaptation]
Order   = [:type]
```
