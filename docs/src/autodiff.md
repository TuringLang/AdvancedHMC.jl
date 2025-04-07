# Gradient in AdvancedHMC.jl

AdvancedHMC.jl supports automatic differentiation using [`LogDensityProblemsAD`](https://github.com/tpapp/LogDensityProblemsAD.jl) across various AD backends and allows user-specified gradients. While the default AD backend for AdvancedHMC.jl is ForwardDiff.jl, we can seamlessly change to other backend like Mooncake.jl using various syntax like `Hamiltonian(metric, ℓπ, AutoMooncake(; config = nothing))`. Different AD backend can also be pluged in using `Hamiltonian(metric, ℓπ, Zygote)`, `Hamiltonian(metric, ℓπ, Val(:Zygote))` but we recommend using ADTypes since that would allow you to have more freedom for specifying the AD backend.

```julia
using AdvancedHMC, DifferentiationInterface, Mooncake, Zygote
hamiltonian = Hamiltonian(metric, ℓπ, AutoMooncake(; config=nothing))
hamiltonian = Hamiltonian(metric, ℓπ, Zygote)
hamiltonian = Hamiltonian(metric, ℓπ, Val{:Zygote})
```

In order to use user-specified gradients, please replace ForwardDiff.jl with `ℓπ_grad` in the `Hamiltonian` constructor as `Hamiltonian(metric, ℓπ, ℓπ_grad)`, where the gradient function `ℓπ_grad` should return a tuple containing both the log-posterior and its gradient, for example `ℓπ_grad(x) = (log_posterior, grad)`.
