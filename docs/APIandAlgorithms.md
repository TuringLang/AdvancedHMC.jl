# API and HMC Algorithms

An important design goal of AdvancedHMC.jl is modularity. This modularity means that different HMC 
variants can be easily constructed by composing various components, such as preconditioning metric (i.e. mass matrix), 
leapfrog integrators, trajectories (static or dynamic), and adaption schemes etc. 

### Hamiltonian mass matrix (`metric`)

The kinetic energy function is parameterized by a choice of a symmetric, positive-definite matrix known as a *mass matrix*.
The inverse of the *mass matrix* is associated with the covariance of the target distribution and hence can be used to pass in correlation assumptions.
AdvancedHMC supports the following choice of `metric`
  
- Unit metric (`UnitEuclideanMetric(dim)`): diagonal matrix of ones  
- Diagonal metric (`DiagEuclideanMetric(dim)`): diagonal matrix with positive diagonal entries
- Dense metric (`DenseEuclideanMetric(dim)`): dense, symmetric positive definite matrix

where `dim` is the dimensionality of the sampling space.

### Integrator (`integrator`)

- Ordinary leapfrog integrator: `Leapfrog(ϵ)`
- Jittered leapfrog integrator with jitter rate `n`: `JitteredLeapfrog(ϵ, n)`
- Tempered leapfrog integrator with tempering rate `a`: `TemperedLeapfrog(ϵ, a)`

where `ϵ` is the step size of leapfrog integration.

### Proposal (`proposal`)

- Static HMC with a fixed number of steps (`n_steps`) (Neal, R. M. (2011)): `StaticTrajectory(integrator, n_steps)`
- HMC with a fixed total trajectory length (`trajectory_length`) (Neal, R. M. (2011)): `HMCDA(integrator, trajectory_length)` 
- Original NUTS with slice sampling (Hoffman, M. D., & Gelman, A. (2014)): `NUTS{SliceTS,ClassicNoUTurn}(integrator)`
- Generalised NUTS with slice sampling (Betancourt, M. (2017)): `NUTS{SliceTS,GeneralisedNoUTurn}(integrator)`
- Original NUTS with multinomial sampling (Betancourt, M. (2017)): `NUTS{MultinomialTS,ClassicNoUTurn}(integrator)`
- Generalised NUTS with multinomial sampling (Betancourt, M. (2017)): `NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)`

### Adaptor (`adaptor`)

- Adapt the mass matrix `metric` of the Hamiltonian dynamics: `mma = MassMatrixAdaptor(metric)`
  - This is lowered to `UnitMassMatrix`, `WelfordVar` or `WelfordCov` based on the type of the mass matrix `metric`
- Adapt the step size of the leapfrog integrator `integrator`: `ssa = StepSizeAdaptor(δ, integrator)`
  - It uses Nesterov's dual averaging with `δ` as the target acceptance rate.
- Combine the two above *naively*: `NaiveHMCAdaptor(mma, ssa)`
- Combine the first two using Stan's windowed adaptation: `StanHMCAdaptor(mma, ssa)`