using Pkg

using Documenter
using AdvancedHMC

# cp(joinpath(@__DIR__, "../README.md"), joinpath(@__DIR__, "src/index.md"))

makedocs(; sitename="AdvancedHMC", format=Documenter.HTML(), warnonly=[:cross_references])
