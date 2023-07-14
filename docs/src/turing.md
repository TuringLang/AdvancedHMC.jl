## Using AdvancedHMC with Turing

In many cases users might want to using a probabilistic programming language such as `Turing.jl` to define a log-likelihood and then use `AdvancedHMC` as a sampling backend.

In order to show how this can be done let us consider a Neal's funnel model:

```julia
using AdvancedHMC, Turing

d = 7
@model function funnel()
    θ ~ Truncated(Normal(0, 3), -3, 3)
    z ~ MvNormal(zeros(d-1), exp(θ)*I)
    x ~ MvNormal(z, I)
end

Random.seed!(1)
(;x) = rand(funnel() | (θ=0,))
cond_model = funnel() | (;x)
```

Now we can simply create a NUTS sampler with `AdvancedHMC` and sample it:

```julia
spl = AdvancedHMC.NUTS(n_adapts=1_000, δ=0.95)
samples = sample(cond_funnel, externalsampler(spl), 50_000;
  progress=true, save_state=true)
```
Note that at the moment the interface between `Turing` and external samplers requires to wrap samplers of the type `AbstractMCMC.AbstractSampler` in `Turing.externalsampler` for them to be interpreted correctly. 