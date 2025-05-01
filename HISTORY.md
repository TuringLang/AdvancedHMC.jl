# AdvancedHMC Changelog

## 0.8.0

  - To make an MCMC transtion from phasepoint `z` using trajectory `τ`(or HMCKernel `κ`) under Hamiltonian `h`, use `transition(h, τ, z)` or `transition(rng, h, τ, z)`(if using HMCKernel, use `transition(h, κ, z)` or `transition(rng, h, κ, z)`).

## v0.7.1

  - README has been simplified, many docs transfered to docs: https://turinglang.org/AdvancedHMC.jl/dev/.
  - ADTypes.jl can be used for specifying the AD backend in `Hamiltonian(metric, ℓπ, AutoForwardDiff())`.
  - SimpleUnpack.jl and Requires.jl are removed from the dependency.
  - `find_good_stepsize` now has fewer allocations.

## v0.7.0

  - Type piracies of Base.rand and Base.randn for vectors of RNGs are removed: Replace `rand(rngs::AbstractVector{<:Random.AbstractRNG})` with `map(rand, rngs)`, `randn(rngs::AbstractVector{<:Random.AbstractRNG})` with `map(randn, rngs)`, `rand(rngs::AbstractVector{<:Random.AbstractRNG}, T, n::Int) (for n == length(rngs))` with `map(Base.Fix2(rand, T), rngs)`, and `randn(rngs::AbstractVector{<:Random.AbstractRNG}, T, m::Int, n::Int) (for n == length(rngs))` with eg `reduce(hcat, map(rng -> randn(rng, T, m), rngs))`.
  - Type piracy `Base.isfinite(x::AbstractVecOrMat)` is removed: Switch to `all(isfinite, x)` if you (possibly implicitly) relied on this definition
  - Abstract fields of `NesterovDualAveraging`, `HMCDA`, `SliceTS`, and `MultinomialTS` are made concrete by adding type parameters: Update occurrences of these types (eg. in function signatures) if necessary
  - Definitions of Base.rand for metrics are removed: Use the (internal) `AdvancedHMC.rand_momentum` function if you depend on this functionality and open an issue to further discuss the API

## v0.5.0

Convenience constructors for common samplers changed to:

  - `HMC(leapfrog_stepsize::Real, n_leapfrog::Int)`
  - `NUTS(target_acceptance::Real)`
  - `HMCDA(target_acceptance::Real, integration_time::Real)`

## v0.2.22

Three functions are renamed.

  - `Preconditioner(metric::AbstractMetric)` -> `MassMatrixAdaptor(metric)` and
  - `NesterovDualAveraging(δ, integrator::AbstractIntegrator)` -> `StepSizeAdaptor(δ, integrator)`
  - `find_good_eps` -> `find_good_stepsize`

## v0.2.15

`n_adapts` is no longer needed to construct `StanHMCAdaptor`; the old constructor is deprecated.

## v0.2.8

Two Hamiltonian trajectory sampling methods are renamed to avoid a name clash with Distributions.

  - `Multinomial` -> `MultinomialTS`
  - `Slice` -> `SliceTS`

## v0.2.0

The gradient function passed to `Hamiltonian` is supposed to return a value-gradient tuple now.
