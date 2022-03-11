using Documenter, TuringWeb
using AdvancedHMC

makedocs(
    sitename = "AdvancedHMC",
    format = Documenter.HTML(),
    modules = [AdvancedHMC]
)

deploydocs(
    repo = "github.com/TuringLang/AdvancedHMC.jl.git"
)
