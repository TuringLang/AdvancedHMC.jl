# A No-U-Turn Sampler (NUTS) Implementation in Julia — Edit

This package implements the No-U-Turn Sampler (NUTS) described in Algorithm 2, 3 and 6 from ([Hoffman & Gelman, 2011][1]).

## Usage

1. Clone this repository in a location (call `PATH`)
2. In order to use this package, add `include("$PATH$/nuts.jl"); using NUTSJulia;`in the top of you file
3. Generate your own initial parameter & log-probaiblity function and pass it with the corresponding setting parameters to the sampler (see API below)

See `example.jl` for an example of sampling from a mixture of Gaussians.

## API

### Naive NUTS (Algorithm 2 in [1])

``
# eff_NUTS(initial_parameter, step_size, log-probablity, number_of_samples)

naive_NUTS(θ0, 0.75, x -> log(f(x)), 2000)
``

### Effective NUTS (Algorithm 3 in [1])

``
# eff_NUTS(initial_parameter, step_size, log-probablity, number_of_samples)
eff_NUTS(θ0, 0.75, x -> log(f(x)), 2000)
``

### NUTS with Dual Averaging (Algorithm 6 in [1])

``
# NUTS(initial_parameter, target_accept_rate, log-probablity, number_of_samples, number_of_adaptations)
NUTS(θ0, 0.65, x -> log(f(x)), 2000, 200)
``

## Reference

[arXiv:1111.4246][1]: Hoffman, M. D., & Gelman, A. (2011, November 18). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. arXiv.org.

[1]: http://arxiv.org/abs/1111.4246
