using Random, LinearAlgebra, ReverseDiff, VecTargets

# Fisher information metric
function gen_∂G∂x(Vfunc, x; f=identity)
    Hfunc = VecTargets.gen_hess(Vfunc, ReverseDiff.track.(x))
    # QUES What's the best output format of this function?
    return x -> ReverseDiff.jacobian(x -> f(Hfunc(x)[3]), x) # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

function reshape_∂G∂x(H)
    d = size(H, 2)
    return cat((H[(i-1)*d+1:i*d,:] for i in 1:d)...; dims=3)
end

function prepare_sample_target(rng, hps, target)
    θ₀ = rand(rng, dim(target))
    
    ℓπ = VecTargets.gen_logpdf(target)
    ∂ℓπ∂θ = VecTargets.gen_logpdf_grad(target, θ₀)
    
    Vfunc = x -> -logpdf(target, x) # potential energy is the negative log-probability
    _Hfunc = VecTargets.gen_hess(Vfunc, θ₀) # x -> (value, gradient, hessian)
    Hfunc = x -> copy.(_Hfunc(x))
    
    fstabilize = H -> H + hps.λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        any(.!(isfinite.(H))) ? diagm(ones(length(x))) : H
    end
    _∂G∂θfunc = gen_∂G∂x(Vfunc, θ₀; f=fstabilize) # size==(4, 2)
    ∂G∂θfunc = x -> reshape_∂G∂x(copy(_∂G∂θfunc(x))) # size==(2, 2, 2)

    return θ₀, ℓπ, ∂ℓπ∂θ, Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end

function prepare_sample(hps; rng=MersenneTwister(1110))
    target = hps.target == :gaussian  ? HighDimGaussian(2) :
             hps.target == :banana    ? Banana() :
             hps.target == :funnel    ? Funnel() :
             hps.target == :funnel101 ? Funnel(101) :
             hps.target == :spiral    ? Spiral(8, 0.1) :
             hps.target == :mogs      ? TwoDimGaussianMixtures() :
             @error "Unknown target $(hps.target)"
    
    θ₀, ℓπ, ∂ℓπ∂θ, _, _, Gfunc, ∂G∂θfunc = prepare_sample_target(rng, hps, target)

    D = dim(target)
    metric = hps.metric == :dense_euclidean          ? DenseEuclideanMetric(D) :
             hps.metric == :dense_riemannian         ? DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc) :
             hps.metric == :dense_riemannian_softabs ? DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, SoftAbsMap(hps.α)) :
             @error "Unknown metric $(hps.metric)"
    kinetic = GaussianKinetic()

    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    TS = EndPointTS
    
    integrator = hps.integrator == :lf  ? Leapfrog(hps.ϵ) :
                 hps.integrator == :glf ? GeneralizedLeapfrog(hps.ϵ, hps.n) :
                 @error "Unknown integrator $(hps.integrator)"

    tc = FixedNSteps(hps.L)
    
    proposal = HMCKernel(Trajectory{TS}(integrator, tc))
    
    return (; rng, hamiltonian, proposal, θ₀)
end

function sample_target(hps; rng=MersenneTwister(1110))    
    rng, hamiltonian, proposal, θ₀ = prepare_sample(hps; rng=rng)

    samples, stats = sample(
        rng, hamiltonian, proposal, θ₀, hps.n_samples; progress=false, verbose=true
    )
    
    return (; target, hamiltonian, samples, stats)
end