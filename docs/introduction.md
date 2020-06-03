# Introduction

AdvancedHMC.jl provides a robust, modular and efficient implementation of advanced HMC algorithms. 
AdvancedHMC.jl provides a standalone implementation of NUTS that recovers and extends Stan’s NUTS algorithm. 
AHMC is written in Julia, a modern high-level language for scientific computing, benefiting from optional 
hardware acceleration and interoperability with a wealth of existing software written in both Julia and other 
languages, such as Python.

It is desirable to have a high quality, standalone NUTS implementation in a high-level language,
for research on HMC algorithms, reproducible comparisons and real-world approximate 
inference applications in different domains. To this end, we introduce
AdvancedHMC.jl (AHMC), a robust, modular and efficient implementation of Stan’s NUTS
and several other commonly used HMC variants in Julia.

## Getting started 

#### A minimal example - sampling from a multivariate Gaussian using NUTS

```julia
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
 proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
 adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
 
 # Run the sampler to draw samples from the specified Gaussian, where
 #   - `samples` will store the samples
 #   - `stats` will store diagnostic statistics for each sample
 samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
```