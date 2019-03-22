# Efficient HMC implementations in Julia

**The code from this repository is used to implement HMC in [Turing.jl](https://github.com/yebai/Turing.jl). Try it out when it's available!**

## Minimal examples - sampling from a multivariate Gaussian using NUTS

```julia
using Distributions, AdvancedHMC
using ForwardDiff: gradient

# Define the target distribution and its gradient
D = 10
logπ(θ::AbstractVector{T}) where {T<:Real} = logpdf(MvNormal(zeros(D), ones(D)), θ)
∂logπ∂θ = θ -> gradient(logπ, θ)

# Sampling parameter settings
ϵ = 0.02
n_steps = 20
n_samples = 100_000

# Initial points
θ_init = randn(D)

# Define metric space, Hamiltonian and sampling method
metric = UnitEuclideanMetric(θ_init)
h = Hamiltonian(metric, logπ, ∂logπ∂θ)
prop = SliceNUTS(Leapfrog(find_good_eps(h, θ_init)))

# Sampling
samples = AdvancedHMC.sample(h, prop, θ_init, n_samples)
```

## Reference

1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([pdf](https://arxiv.org/pdf/1206.1901))

2. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

3. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2), 123-214. ([link](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x))

4. Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

5. Betancourt, M. (2016). Identifying the optimal integration time in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1601.00225](https://arxiv.org/abs/1601.00225).

6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([link][1])


[1]: http://arxiv.org/abs/1111.4246
