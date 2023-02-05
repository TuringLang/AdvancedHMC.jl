using Pkg

using Documenter
using AdvancedHMC

# cp(joinpath(@__DIR__, "../README.md"), joinpath(@__DIR__, "src/index.md"))

makedocs(sitename = "AdvancedHMC", format = Documenter.HTML(), modules = [AdvancedHMC])

deploydocs(
    repo = "github.com/TuringLang/AdvancedHMC.jl.git",
    push_preview = true, # allow PR to deploy
)
