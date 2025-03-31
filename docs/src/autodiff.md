# Gradient in AdvancedHMC.jl

AdvancedHMC.jl supports automatic differentiation using [`LogDensityProblemsAD`](https://github.com/tpapp/LogDensityProblemsAD.jl) and user-specified gradients. While the default AD backend for AdvancedHMC.jl is ForwardDiff.jl, we can seamlessly change to other backend like Zygote.jl using various syntax like `Hamiltonian(metric, ℓπ, Zygote)`, `Hamiltonian(metric, ℓπ, Val(:Zygote))` or via ADTypes.jl `Hamiltonian(metric, ℓπ, AutoZygote())`.

In order to use user-specified gradients, please replace ForwardDiff.jl with `ℓπ_grad` in the `Hamiltonian` constructor, where the gradient function `ℓπ_grad` should return a tuple containing both the log-posterior and its gradient.
