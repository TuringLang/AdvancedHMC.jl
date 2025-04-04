# AdvancedHMC.jl

[![CI](https://github.com/TuringLang/AdvancedHMC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/TuringLang/AdvancedHMC.jl/actions/workflows/CI.yml)
[![DOI](https://zenodo.org/badge/72657907.svg)](https://zenodo.org/badge/latestdoi/72657907)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AdvancedHMC.jl/badge.svg?branch=kx%2Fbug-fix)](https://coveralls.io/github/TuringLang/AdvancedHMC.jl?branch=kx%2Fbug-fix)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/AdvancedHMC.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/AdvancedHMC.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

AdvancedHMC.jl provides a robust, modular, and efficient implementation of advanced Hamiltonian Monte Carlo(HMC) algorithms. AdvancedHMC.jl is part of [Turing.jl](https://github.com/TuringLang/Turing.jl), a probabilistic programming library in Julia.
If you are interested in using AdvancedHMC.jl through a probabilistic programming language, please check it out!

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

If you using AdvancedHMC.jl directly through Turing.jl, please consider citing the following publication:

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

 2. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([arXiv](http://arxiv.org/abs/1111.4246))
 3. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
 4. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2), 123-214. ([arXiv](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x))
 5. Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).
 6. Betancourt, M. (2016). Identifying the optimal integration time in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1601.00225](https://arxiv.org/abs/1601.00225).
 7. 
