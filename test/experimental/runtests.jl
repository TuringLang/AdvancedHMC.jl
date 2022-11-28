using ReTest

include("relativistic_hmc.jl")

@main function runtests(patterns...; dry::Bool=false)
    retest(patterns...; dry=dry, verbose=Inf)
end