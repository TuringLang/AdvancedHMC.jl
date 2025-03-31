using Pkg

using Documenter, DocumenterCitations
using AdvancedHMC

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    sitename="AdvancedHMC",
    format=Documenter.HTML(;
        assets=["assets/favicon.ico"],
        canonical="https://turinglang.org/AdvancedHMC.jl/stable/",
    ),
    warnonly=[:cross_references],
    plugins=[bib],
    pages=[
        "AdvancedHMC.jl" => "index.md",
        "Get Started" => "get_started.md",
        "Automatic Differentiation Backends" => "autodiff.md",
        "Detailed API" => "api.md",
        "Change Log" => "changelog.md",
        "References" => "references.md",
    ],
)
