using AdvancedHMC, ForwardDiff, Zygote
using LogDensityProblems
using LinearAlgebra
using Plots
using MCMCDiagnosticTools

D = 2

function basis_vector(i::Int)
    v = zeros(D)
    v[i] = 1.0
    return v
end
step_functions = [
    LinearStepFunction(sign * basis_vector(idx), 1.0)
    for sign in [-1.0, 1.0]
    for idx in 1:D
]

# Sidestep AD for now by manually defining log density and score

sigma = 1 / 5

function ℓπ(θ, signature)
    if all(signature)
        return -sum(abs2, θ) / (2 * sigma)
    else
        return -Inf
    end
end
function ℓπ(θ)
    signature = [
        evaluate(s, θ)
        for s in step_functions
    ]
    return ℓπ(θ, signature)
end

function ∂ℓπ∂θ(θ, signature)
    if all(signature)
        return ℓπ(θ, signature), -θ / sigma
    else
        return -Inf, zeros(D)
    end
end
function ∂ℓπ∂θ(θ)
    signature = [
        evaluate(s, θ)
        for s in step_functions
    ]
    return ∂ℓπ∂θ(θ, signature)
end

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 2_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = DiscontinuousHamiltonian(metric, ℓπ, ∂ℓπ∂θ, step_functions)

# Define a leapfrog solver, with initial step size chosen heuristically
# initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
initial_ϵ = 0.1
integrator = DiscontinuousLeapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
# proposal = HMCDA(integrator, 1.0)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
# adaptor = StepSizeAdaptor(0.8, integrator)
# adaptor = NoAdaptation()

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
nchains = 8
println("Starting sampling...")
elapsed = @elapsed begin
    chains = Vector{Any}(undef, nchains)
    Threads.@threads for i in 1:nchains
        initial_θ = rand(D)
        samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; verbose=false)
        chains[i] = mapreduce(permutedims, vcat, samples)
    end
end
println("Sampling finished.")

# Plot samples
combined_samples = mapreduce(permutedims, hcat, chains)
plot(combined_samples[1, :], combined_samples[2, :], seriestype=:scatter, markersize=2, markerstrokewidth=0, markeralpha=0.1)
plot!(xlims=(-1.1, 1.1), ylims=(-1.1, 1.1), legend=false)
plot!(size=(600, 600), aspect_ratio=:equal)

samples_array = cat(chains..., dims=3)
samples_array = permutedims(samples_array, (1, 3, 2))
println(size(samples_array))
ess, rhat = ess_rhat(samples_array)
println("ESSs: ", ess ./ (n_samples * nchains))
min_ess = minimum(ess) / (n_samples * nchains)
println("Minumum ESS: ", min_ess)
println("Elapsed: ", elapsed)
println("s/minESS: ", elapsed / min_ess)
