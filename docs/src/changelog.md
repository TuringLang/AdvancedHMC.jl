**CHANGELOG**

  - [v0.5.0] **Breaking!** Convenience constructors for common samplers changed to:
    
      + `HMC(leapfrog_stepsize::Real, n_leapfrog::Int)`
      + `NUTS(target_acceptance::Real)`
      + `HMCDA(target_acceptance::Real, integration_time::Real)`

  - [v0.2.22] Three functions are renamed.
    
      + `Preconditioner(metric::AbstractMetric)` -> `MassMatrixAdaptor(metric)` and
      + `NesterovDualAveraging(δ, integrator::AbstractIntegrator)` -> `StepSizeAdaptor(δ, integrator)`
      + `find_good_eps` -> `find_good_stepsize`
  - [v0.2.15] `n_adapts` is no longer needed to construct `StanHMCAdaptor`; the old constructor is deprecated.
  - [v0.2.8] Two Hamiltonian trajectory sampling methods are renamed to avoid a name clash with Distributions.
    
      + `Multinomial` -> `MultinomialTS`
      + `Slice` -> `SliceTS`
  - [v0.2.0] The gradient function passed to `Hamiltonian` is supposed to return a value-gradient tuple now.
