using AdvancedHMC, PosteriorDB, StanLogDensityProblems, Random

if !@isdefined pdb 
    const pdb = PosteriorDB.database()
end
stan_problem(path, data) = StanProblem(
    path, data;
    nan_on_error=true,
    make_args=["STAN_THREADS=TRUE"],
    warn=false
)
stan_problem(posterior_name::AbstractString) = stan_problem(
    PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(PosteriorDB.posterior(pdb, (posterior_name))), "stan")), 
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, (posterior_name))), String)
)

begin
lpdf = stan_problem("radon_mn-radon_county_intercept")

n_adapts = n_samples = 1000
rng = Xoshiro(1)
spl = NUTS(0.8)
initial_params = nothing
model = AdvancedHMC.AbstractMCMC._model(lpdf)
(;logdensity) = model
metric = AdvancedHMC.make_metric(spl, logdensity)
hamiltonian = AdvancedHMC.Hamiltonian(metric, model)
initial_params = AdvancedHMC.make_initial_params(rng, spl, logdensity, initial_params)
ϵ = AdvancedHMC.make_step_size(rng, spl, hamiltonian, initial_params)
integrator = AdvancedHMC.make_integrator(spl, ϵ)
κ = AdvancedHMC.make_kernel(spl, integrator)
# adaptor = AdvancedHMC.make_adaptor(spl, metric, integrator)
adaptor = AdvancedHMC.StanHMCAdaptor(
    AdvancedHMC.Adaptation.NutpieVar(size(metric); var=copy(metric.M⁻¹)), 
    AdvancedHMC.StepSizeAdaptor(spl.δ, integrator)
)
h, t = AdvancedHMC.sample_init(rng, hamiltonian, initial_params)
# Using the below uses Nutpie (as in position and gradients)
initial_state = AdvancedHMC.HMCState(0, t, metric, κ, adaptor)
# Using the below uses Stan (as in only positions)
# initial_state = nothing
@time samples = AdvancedHMC.sample(
    rng,
    model,
    spl,
    n_adapts + n_samples;
    n_adapts=n_adapts, initial_state,
    progress=true, 
)
nothing
end
