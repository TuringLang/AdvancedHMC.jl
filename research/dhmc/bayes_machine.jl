using AdvancedHMC, ForwardDiff, Zygote
using LogDensityProblems
using LinearAlgebra
using Plots
using MCMCDiagnosticTools

X = [
    -1.0 0.0;
    -1.0 1.0;
    1.0 0.0;
    0.8 0.2;
    0.2 -0.3;
]
y = [-1.0, -1.0, 1.0, 1.0, 1.0]

weight_std = 1.0
bias_std = 2.0

struct LogTargetDensity
    dim::Int
end
function LogDensityProblems.logdensity(p::LogTargetDensity, θ)
    w, b = θ[1:end-1], θ[end]
    # w, b = θ, 0.0
    if all(y .* (X * w .+ b) .>= 0.0)
        -sum(abs2, w) / (2 * weight_std^2) - b^2 / (2 * bias_std^2)
    else
        -Inf
    end
end
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Choose parameter dimensionality and initial parameter value
D = size(X, 2) + 1
ℓπ = LogTargetDensity(D)

# Plot log density for fixed bias = 0.0
# ws = range(-2.0, 2.0; length=100)
# grid_ws = repeat(ws, 1, 100), repeat(ws, 1, 100)'
# log_density = [LogDensityProblems.logdensity(ℓπ, [w1, w2, 0.0]) for (w1, w2) in zip(grid_ws...)];
# # log_density = [LogDensityProblems.logdensity(ℓπ, [w1, w2]) for (w1, w2) in zip(grid_ws...)];
# heatmap(ws, ws, log_density'; xlabel="w1", ylabel="w2", title="log density for fixed bias = 0.0")

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 2_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
# initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
initial_ϵ = 0.1
constraints = [
    LinearConstraint([yi .* Xi; yi], 0.0)
    # LinearConstraint(yi .* Xi, 0.0)
    for (Xi, yi) in zip(eachrow(X), y)
]
integrator = ConstrainedLeapfrog(initial_ϵ, constraints)


# Plot feasible region
# ws = range(-2.0, 2.0; length=100)
# grid_ws = repeat(ws, 1, 100), repeat(ws, 1, 100)'
# feasible = sum([[dot(c.a, [w1, w2, 0.0]) >= 0 for (w1, w2) in zip(grid_ws...)] for c in constraints]) .>= 4;
# # log_density = [LogDensityProblems.logdensity(ℓπ, [w1, w2]) for (w1, w2) in zip(grid_ws...)];
# heatmap(ws, ws, feasible'; xlabel="w1", ylabel="w2", title="log density for fixed bias = 0.0")

proposal = HMCDA(integrator, 1.0)
adaptor = AdvancedHMC.NoAdaptation()
# proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
# adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
nchains = 1
println("Starting sampling...")
elapsed = @elapsed begin
    chains = Vector{Any}(undef, nchains)
    # Threads.@threads for i in 1:nchains
    for i in 1:nchains
        initial_θ = nothing
        while true
            initial_θ = randn(D)
            w, b = initial_θ[1:end-1], initial_θ[end]
            # w, b = initial_θ, 0.0
            if all(y .* (X * w .+ b) .>= 0.0)
                break
            end
        end
        samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; verbose=false)
        chains[i] = mapreduce(permutedims, vcat, samples)
    end
end
println("Sampling finished.")

# Plot samples
combined_samples = mapreduce(permutedims, hcat, chains)
plot(combined_samples[1, :], combined_samples[2, :], seriestype=:scatter, markersize=2, markerstrokewidth=0, markeralpha=0.1)
plot!(xlims=(-3.0, 3.0), ylims=(-3.0, 3.0), legend=false)
plot!(size=(600, 600), aspect_ratio=:equal)
# TODO: add utility for plotting constraints

# Plot posterior
xs = range(-2.0, 2.0, length=100)
ys = range(-2.0, 2.0, length=100)
# Mesh grid
mxs, mys = repeat(xs, 1, 100), repeat(ys, 1, 100)'
zs = zeros(100, 100)
for i in 1:100
    for j in 1:100
        zs[i, j] = sum(combined_samples' * [mxs[i, j], mys[i, j], 1.0] .>= 0) / (n_samples * nchains)
        # zs[i, j] = sum(combined_samples' * [mxs[i, j], mys[i, j]] .>= 0) / (n_samples * nchains)
    end
end

# Heatmap
p = heatmap(xs, ys, zs', aspect_ratio=:equal, legend=false, color=:redsblues, clim=(0.0, 1.0))

# Colour markers
colours = [yi == 1.0 ? "blue" : "red" for yi in y]
plot!(
    p, X[:, 1], X[:, :2],
    seriestype=:scatter, markersize=5,
    markerstrokewidth=1, color=colours,
    markerstrokecolor=:white
);
plot!(p, xlims=(-2.0, 2.0), ylims=(-2.0, 2.0), legend=false);
display(p);
