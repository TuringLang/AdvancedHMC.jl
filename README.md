# Efficient HMC implementations in Julia

[![Build Status](https://travis-ci.org/TuringLang/AdvancedHMC.jl.svg?branch=master)](https://travis-ci.org/TuringLang/AdvancedHMC.jl) [![Coverage Status](https://coveralls.io/repos/github/TuringLang/AdvancedHMC.jl/badge.svg?branch=kx%2Fbug-fix)](https://coveralls.io/github/TuringLang/AdvancedHMC.jl?branch=kx%2Fbug-fix)

**The code from this repository is used to implement HMC samplers in [Turing.jl](https://github.com/TuringLang/Turing.jl).**

**NEWS**
- We presented a poster for AdvancedHMC.jl at StanCon 2019. ([pdf](https://github.com/TuringLang/AdvancedHMC.jl/files/3730367/StanCon-AHMC.pdf))

**API CHANGE**
- [v0.2.8] Two exported types are renamed: `Multinomial` -> `MultinomialTS` and `Slice` -> `SliceTS`.
- [v0.2.0] The gradient function passed to `Hamiltonian` is supposed to return a value-gradient tuple now.

## Minimal examples - sampling from a multivariate Gaussian using NUTS

```julia
### Define the target distribution and its gradient
using Distributions: logpdf, MvNormal
using DiffResults: GradientResult, value, gradient
using ForwardDiff: gradient!

D = 10
target = MvNormal(zeros(D), ones(D))
ℓπ(θ) = logpdf(target, θ)

function ∂ℓπ∂θ(θ)
    res = GradientResult(θ)
    gradient!(res, ℓπ, θ)
    return (value(res), gradient(res))
end

### Build up a HMC sampler to draw samples
using AdvancedHMC

# Sampling parameter settings
n_samples, n_adapts = 10_000, 2_000

# Draw a random starting points
θ_init = randn(D)

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(D)
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
int = Leapfrog(find_good_eps(h, θ_init))
prop = NUTS{MultinomialTS,GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, int))

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples, stats = sample(h, prop, θ_init, n_samples, adaptor, n_adapts; progress=true)
```

## Reference

1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([pdf](https://arxiv.org/pdf/1206.1901))

2. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

3. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2), 123-214. ([link](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x))

4. Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

5. Betancourt, M. (2016). Identifying the optimal integration time in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1601.00225](https://arxiv.org/abs/1601.00225).

6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([link][1])


[1]: http://arxiv.org/abs/1111.4246
