# A No-U-Turn Sampler (NUTS) Implementation in Julia

**The code from this repository is already adapted to be used to implement NUTS in [Turing.jl](https://github.com/yebai/Turing.jl). Try it out!**

This package implements the No-U-Turn Sampler (NUTS) described in Algorithm 2, 3 and 6 from ([Hoffman & Gelman, 2011][1]).

## Usage

1. Clone this repository in a location (call `PATH`)
2. In order to use this package, add `include("$PATH/nuts.jl"); using NUTSJulia;`in the top of you file
3. Generate your own initial parameter & log-probaiblity function and pass it with the corresponding setting parameters to the sampler (see API below)

See `example.jl` for an example of sampling from a mixture of Gaussians.

## API

### Naive NUTS - Algorithm 2 in ([Hoffman & Gelman, 2011][1])

```
# eff_NUTS(initial_θ, step_size, log_pdf, sample_num)
naive_NUTS(θ0, 0.75, x -> log(f(x)), 2000)
```

### Effective NUTS - Algorithm 3 in ([Hoffman & Gelman, 2011][1])

```
# eff_NUTS(initial_θ, step_size, log_pdf, sample_num)
eff_NUTS(θ0, 0.75, x -> log(f(x)), 2000)
```

### NUTS with Dual Averaging - Algorithm 6 in ([Hoffman & Gelman, 2011][1])

```
# NUTS(initial_θ, target_accept_rate, log_pdf, sample_num, adapt_num)
NUTS(θ0, 0.65, x -> log(f(x)), 2000, 200)
```

## Reference

[arXiv:1111.4246][1]: Hoffman, M. D., & Gelman, A. (2011, November 18). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. arXiv.org.

[1]: http://arxiv.org/abs/1111.4246
