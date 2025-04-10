# AdvancedHMC.jl

[![CI](https://github.com/TuringLang/AdvancedHMC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/TuringLang/AdvancedHMC.jl/actions/workflows/CI.yml)
[![DOI](https://zenodo.org/badge/72657907.svg)](https://zenodo.org/badge/latestdoi/72657907)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AdvancedHMC.jl/badge.svg?branch=main)](https://coveralls.io/github/TuringLang/AdvancedHMC.jl?branch=main)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/AdvancedHMC.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/AdvancedHMC.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

**AdvancedHMC.jl** provides robust, modular, and efficient implementation of advanced Hamiltonian Monte Carlo (HMC) algorithms in Julia. It is a backend for probabilistic programming languages like [Turing.jl](https://github.com/TuringLang/Turing.jl), but can also be used directly for flexible MCMC sampling when fine-grained control is desired.

**Key Features**

  - Implementation of state-of-the-art [HMC variants](https://turinglang.org/AdvancedHMC.jl/dev/api/) (e.g., NUTS).
  - The modular design allows for the customization of metrics, Hamiltonian trajectory simulation, and adaptation.
  - Integration with the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface for defining target distributions, and [LogDensityProblemsAD.jl](https://github.com/TuringLang/LogDensityProblemsAD.jl) for supporting automatic differentiation backends.
  - Built upon the [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl) interface for MCMC sampling.

## Installation

AdvancedHMC.jl is a registered Julia package. You can install it using the Julia package manager:

```julia
using Pkg
Pkg.add("AdvancedHMC")
```

## Quick Start: Sampling a Multivariate Normal

Here's a basic example demonstrating how to sample from a target distribution (a standard multivariate normal) using the No-U-Turn Sampler (NUTS).

```julia
using AdvancedHMC, AbstractMCMC
using LogDensityProblems, LogDensityProblemsAD, ADTypes # For defining the target distribution & its gradient
using ForwardDiff # An example AD backend
using Random # For initial parameters

# 1. Define the target distribution using the LogDensityProblems interface
struct LogTargetDensity
    dim::Int
end
# Log density of a standard multivariate normal distribution
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(abs2, θ) / 2
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
# Declare that the log density function is defined
function LogDensityProblems.capabilities(::Type{LogTargetDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

# Set parameter dimensionality
D = 10

# 2. Wrap the log density function and specify the AD backend.
#    This creates a callable struct that computes the log density and its gradient.
ℓπ = LogTargetDensity(D)
model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(AutoForwardDiff(), ℓπ))

# 3. Set up the HMC sampler
#    - Use the No-U-Turn Sampler (NUTS)
#    - Specify the target acceptance probability (δ) for step size adaptation
sampler = NUTS(0.8) # Target acceptance probability δ=0.8

# Define the number of adaptation steps and sampling steps
n_adapts, n_samples = 2_000, 1_000

# 4. Run the sampler!
#    We use the AbstractMCMC.jl interface.
#    Provide the model, sampler, total number of steps, and adaptation steps.
#    An initial parameter vector `initial_θ` is also required.
initial_θ = randn(D)

samples = AbstractMCMC.sample(
    Random.default_rng(),
    model,
    sampler,
    n_adapts + n_samples;
    n_adapts=n_adapts,
    initial_params=initial_θ,
    progress=true, # Optional: Show a progress bar
)

# `samples` now contains the MCMC chain. You can analyze it using packages
# like StatsPlots.jl, ArviZ.jl, or MCMCChains.jl.
```

For more advanced usage, please refer to the [docs](https://turinglang.org/AdvancedHMC.jl/dev/get_started/).

## Contributing

Contributions are highly welcome! If you find a bug, have a suggestion, or want to contribute code, please open an issue or pull request.

## License

AdvancedHMC.jl is licensed under the MIT License. See the LICENSE file for details.

## Citing AdvancedHMC.jl

If you use AdvancedHMC.jl for your own research, please consider citing the following publication:

Kai Xu, Hong Ge, Will Tebbutt, Mohamed Tarek, Martin Trapp, Zoubin Ghahramani: "AdvancedHMC.jl: A robust, modular and efficient implementation of advanced HMC algorithms.", *Symposium on Advances in Approximate Bayesian Inference*, 2020. ([abs](http://proceedings.mlr.press/v118/xu20a.html), [pdf](http://proceedings.mlr.press/v118/xu20a/xu20a.pdf))

with the following BibTeX entry:

```
@inproceedings{xu2020advancedhmc,
  title={AdvancedHMC. jl: A robust, modular and efficient implementation of advanced HMC algorithms},
  author={Xu, Kai and Ge, Hong and Tebbutt, Will and Tarek, Mohamed and Trapp, Martin and Ghahramani, Zoubin},
  booktitle={Symposium on Advances in Approximate Bayesian Inference},
  pages={1--10},
  year={2020},
  organization={PMLR}
}
```

If you are using AdvancedHMC.jl directly through Turing.jl, please consider citing the following publication:

Hong Ge, Kai Xu, and Zoubin Ghahramani: "Turing: a language for flexible probabilistic inference.", *International Conference on Artificial Intelligence and Statistics*, 2018. ([abs](http://proceedings.mlr.press/v84/ge18b.html), [pdf](http://proceedings.mlr.press/v84/ge18b/ge18b.pdf))

with the following BibTeX entry:

```
@inproceedings{ge2018turing,
  title={Turing: A language for flexible probabilistic inference},
  author={Ge, Hong and Xu, Kai and Ghahramani, Zoubin},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1682--1690},
  year={2018},
  organization={PMLR}
}
```

## References

 1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))
 2. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
 3. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2), 123-214. ([arXiv](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x))
 4. Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).
 5. Betancourt, M. (2016). Identifying the optimal integration time in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1601.00225](https://arxiv.org/abs/1601.00225).
 6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([arXiv](http://arxiv.org/abs/1111.4246))
