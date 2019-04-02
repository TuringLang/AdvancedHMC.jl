# Efficient HMC implementations in Julia

[![Build Status](https://travis-ci.org/TuringLang/AdvancedHMC.jl.svg?branch=master)](https://travis-ci.org/TuringLang/AdvancedHMC.jl) [![Coverage Status](https://coveralls.io/repos/github/TuringLang/AdvancedHMC.jl/badge.svg?branch=kx%2Fbug-fix)](https://coveralls.io/github/TuringLang/AdvancedHMC.jl?branch=kx%2Fbug-fix)

**The code from this repository is used to implement HMC in [Turing.jl](https://github.com/yebai/Turing.jl). Try it out when it's available!**

## Minimal examples - sampling from a multivariate Gaussian using NUTS

```julia
using Distributions: MvNormal, logpdf
using ForwardDiff: gradient
using AdvancedHMC

# Define the target distribution and its gradient
const D = 10
const target = MvNormal(zeros(D), ones(D))
logπ(θ::AbstractVector{<:Real}) = logpdf(target, θ)
∂logπ∂θ(θ::AbstractVector{<:Real}) = gradient(logπ, θ)

# Sampling parameter settings
n_samples = 100_000
n_adapts = 2_000

# Initial points
θ_init = randn(D)

# Define metric space, Hamiltonian and sampling method
metric = DenseEuclideanMetric(D)
h = Hamiltonian(metric, logπ, ∂logπ∂θ)
prop = NUTS(Leapfrog(find_good_eps(h, θ_init)))
adaptor = StanNUTSAdaptor(n_adapts, PreConditioner(metric), NesterovDualAveraging(0.8, prop.integrator.ϵ))

# Sampling
samples = sample(h, prop, θ_init, n_samples, adaptor, n_adapts)
```

## Reference

1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([pdf](https://arxiv.org/pdf/1206.1901))

2. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

3. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2), 123-214. ([link](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x))

4. Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

5. Betancourt, M. (2016). Identifying the optimal integration time in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1601.00225](https://arxiv.org/abs/1601.00225).

6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([link][1])


[1]: http://arxiv.org/abs/1111.4246
