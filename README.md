# AdvancedHMC.jl: a robust, modular and efficient implementation of advanced HMC algorithms

[![Build Status](https://travis-ci.org/TuringLang/AdvancedHMC.jl.svg?branch=master)](https://travis-ci.org/TuringLang/AdvancedHMC.jl)
[![DOI](https://zenodo.org/badge/72657907.svg)](https://zenodo.org/badge/latestdoi/72657907)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AdvancedHMC.jl/badge.svg?branch=kx%2Fbug-fix)](https://coveralls.io/github/TuringLang/AdvancedHMC.jl?branch=kx%2Fbug-fix)

`AdvancedHMC.jl` is part of [Turing.jl](https://github.com/TuringLang/Turing.jl), a probabilistic programming library in Julia. 
If you are interested in using `AdvancedHMC.jl` through a probabilistic programming language, please check it out!


**NEWS**
- We will present AdvancedHMC.jl at [AABI](http://approximateinference.org/) 2019 in Vancouver, Canada.
- We presented a poster for AdvancedHMC.jl at [StanCon 2019](https://mc-stan.org/events/stancon2019Cambridge/) in Cambridge, UK. ([pdf](https://github.com/TuringLang/AdvancedHMC.jl/files/3730367/StanCon-AHMC.pdf))

**API CHANGES**
- [v0.2.15] `n_adapts` is not needed to construct `StanHMCAdaptor`; the old constructor is deprecated.
- [v0.2.8] Two exported types are renamed: `Multinomial` -> `MultinomialTS` and `Slice` -> `SliceTS`.
- [v0.2.0] The gradient function passed to `Hamiltonian` is supposed to return a value-gradient tuple now.

## A minimal example - sampling from a multivariate Gaussian using NUTS

```julia
### Define the target distribution
using Distributions: logpdf, MvNormal

D = 10
target = MvNormal(zeros(D), ones(D))
ℓπ(θ) = logpdf(target, θ)

### Build up a HMC sampler to draw samples
using AdvancedHMC, ForwardDiff  # or using Zygote
                                # AdvancedHMC will use it for gradient
# Sampling parameter settings
n_samples, n_adapts = 12_000, 2_000

# Draw a random starting points
θ_init = rand(D)

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(D)
h = Hamiltonian(metric, ℓπ) # Hamiltonian(metric, ℓπ, ∂ℓπ∂θ) for hand-coded gradient ∂ℓπ∂θ
int = Leapfrog(find_good_eps(h, θ_init))
prop = NUTS{MultinomialTS,GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, int))

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples, stats = sample(h, prop, θ_init, n_samples, adaptor, n_adapts; progress=true)
```

## API and supported HMC algorithms

An important design goal of `AdvancedHMC.jl` is to be modular, and support algorithmic research on HMC. 
This modularity means that different HMC variants can be easily constructed by composing various components, such as preconditioning metric (i.e. mass matrix), leapfrog integrators,  trajectories (static or dynamic), and adaption schemes etc. 
The minimal example above can be modified to suit particular inference problems by picking components from the list below.

### Preconditioning matrix (`metric`)

- Unit metric: `UnitEuclideanMetric(dim)` 
- Diagonal metric: `DiagEuclideanMetric(dim)`
- Dense metric: `DenseEuclideanMetric(dim)` 

where `dim` is the dimensionality of the sampling space.

### Integrator (`int`)

- Ordinary leapfrog integrator: `Leapfrog(ϵ)`
- Jittered leapfrog integrator with jitter rate `n`: `JitteredLeapfrog(ϵ, n)`
- Tempered leapfrog integrator with tempering rate `a`: `TemperedLeapfrog(ϵ, a)`

where `ϵ` is the step size of leapfrog integration.

### Proposal (`prop`)

- Static HMC with a fixed number of steps (`n_steps`): `StaticTrajectory(int, n_steps)`
- HMC with a fixed total trajectory length (`len_traj`): `HMCDA(int, len_traj)` 
- Original NUTS with slice sampling: `NUTS{SliceTS,ClassicNoUTurn}(int)`
- Generalised NUTS with slice sampling: `NUTS{SliceTS,GeneralisedNoUTurn}(int)`
- Original NUTS with multinomial sampling: `NUTS{MultinomialTS,ClassicNoUTurn}(int)`
- Generalised NUTS with multinomial sampling: `NUTS{MultinomialTS,GeneralisedNoUTurn}(int)`

where `int` is the integrator used.

### Adaptor (`adaptor`)

- Preconditioning on metric space `metric`: `pc = Preconditioner(metric)`
- Nesterov's dual averaging with target acceptance rate `δ` on integrator `int`: `da = NesterovDualAveraging(δ, int)`
- Combine the two above *naively*: `NaiveHMCAdaptor(pc, da)`
- Combine the first two using Stan's windowed adaptation: `StanHMCAdaptor(pc, da)`

All the combinations are tested in [this file](https://github.com/TuringLang/AdvancedHMC.jl/blob/master/test/hmc.jl) except from using tempered leapfrog integrator together with adaptation, which we found unstable empirically.

## The `sample` function signature in detail

```julia
sample(
    rng::AbstractRNG,
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVector{<:AbstractFloat},
    n_samples::Int,
    adaptor::Adaptation.AbstractAdaptor=Adaptation.NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    drop_warmup::Bool=false,
    verbose::Bool=true,
    progress::Bool=false
)
```

Sample `n_samples` samples using the proposal `τ` under Hamiltonian `h`

- The randomness is controlled by `rng`. 
    - If `rng` is not provided, `GLOBAL_RNG` will be used.
- The initial point is given by `θ`.
- The adaptor is set by `adaptor`, for which the default is no adaptation.
    - It will perform `n_adapts` steps of adaptation, for which the default is the minimum of `1_000` and 10% of `n_samples`.
- `drop_warmup` specifies whether to drop samples.
- `verbose` controls the verbosity.
- `progress` controls whether to show the progress meter or not.

## Citing AdvancedHMC.jl ##
If you use **AdvancedHMC.j** for your own research, please consider citing the following publication: Hong Ge, Kai Xu, and Zoubin Ghahramani: **Turing: a language for flexible probabilistic inference.** AISTATS 2018 [pdf](http://proceedings.mlr.press/v84/ge18b.html) [bibtex](https://github.com/TuringLang/Turing.jl/blob/master/CITATION.bib)

## References

1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))

2. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

3. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2), 123-214. ([arXiv](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x))

4. Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

5. Betancourt, M. (2016). Identifying the optimal integration time in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1601.00225](https://arxiv.org/abs/1601.00225).

6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([arXiv](http://arxiv.org/abs/1111.4246))
