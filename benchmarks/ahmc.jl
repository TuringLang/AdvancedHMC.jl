include(joinpath(@__DIR__, "argparse.jl"))

using AdvancedHMC, ForwardDiff, Zygote, ReverseDiff

const ADDICT = Dict(
    "forwarddiff" => ForwardDiff,
    "zygote"      => Zygote,
    "reversediff" => ReverseDiff,
)

const T = Float32

target_dir = joinpath(@__DIR__, "targets")
for fname in readdir(target_dir)
    include(joinpath(target_dir, fname))
end

const TARGETDICT = Dict(
    "gaussian" => ℓπ_gaussian,
    "mog"      => ℓπ_mog,
    "gdemo"    => ℓπ_gdemo,
)

function run_hmc(n_samples::Int, n_chains::Int, n_dims::Int, ℓπ, ∂ℓπ∂θ; ϵ=0.01, n_steps=4)
    initial_θ = n_chains == 1 ? zeros(T, n_dims) : zeros(T, n_dims, n_chains)
    metric = UnitEuclideanMetric(T, size(initial_θ))
    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    integrator = Leapfrog(T(ϵ))
    proposal = StaticTrajectory(integrator, n_steps)
    samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples; progress=false, verbose=false)
end

function main()
    parsed_args = parse_commandline()
    @info "Args" parsed_args...
    n_dims_list   = parsed_args[:n_dims]
    n_samples     = parsed_args[:n_samples]
    n_chains_list = parsed_args[:n_chains]
    n_runs        = parsed_args[:n_runs]
    target        = parsed_args[:target]
    ad            = parsed_args[:ad]

    if target == "gdemo"
        @assert n_dims_list == [2] "gdemo only supports n_dims = 2."
    end
    ℓπ = TARGETDICT[target]

    if ad == "hand"
        @assert target == "gaussian" "Only gaussian supports hand coded gradient"
        ∂ℓπ∂θ = ∂ℓπ∂θ_gaussian
    else
        ∂ℓπ∂θ = ADDICT[ad]
    end

    function run_hmc_on_target(ns, nc, nd)
        return run_hmc(ns, nc, nd, ℓπ, ∂ℓπ∂θ)
    end

    for i_run in 1:n_runs
        println("Running $i_run ...")
        ts = Matrix{Float64}(undef, length(n_dims_list), length(n_chains_list))
        for (i, nd) in enumerate(n_dims_list), (j, nc) in enumerate(n_chains_list)
            run_hmc_on_target(1, nc, nd)  # to get rid of compilation time
            print("  n_dims=$nd, n_chains=$nc ...")
            val, t, bytes, gctime, memallocs = @timed run_hmc_on_target(n_samples, nc, nd)
            println(" Done! $(lpad(round(t; digits=3), 7)) seconds used")
            ts[i,j]  =t
        end
        println(ts)
    end
end

main()
