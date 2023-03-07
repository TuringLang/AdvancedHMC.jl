using Random, LinearAlgebra, ReverseDiff, VecTargets

# Fisher information metric
function gen_∂H∂x(func, x; f=identity)
    hess = VecTargets.gen_hess(func, ReverseDiff.track.(x))
    # QUES What's the best output format of this function?
    return x -> ReverseDiff.jacobian(x -> f(hess(x)[3]), x) # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

function reshape_∂H∂x(H)
    d = size(H, 2)
    return cat((H[(i-1)*d+1:i*d,:] for i in 1:d)...; dims=3)
end

function prepare_sample(hps; rng=MersenneTwister(1110))
    target = hps.target == :gaussian  ? HighDimGaussian(2) :
             hps.target == :banana    ? Banana() :
             hps.target == :funnel    ? Funnel() :
             hps.target == :funnel101 ? Funnel(101) :
             hps.target == :spiral    ? Spiral(8, 0.1) :
             hps.target == :mogs      ? TwoDimGaussianMixtures() :
             @error "Unknown target $(hps.target)"
    D = dim(target)
    initial_θ = rand(rng, D)
    
    ℓπ = x -> logpdf(target, x)
    _∂ℓπ∂θ = gen_logpdf_grad(target, initial_θ)
    ∂ℓπ∂θ = x -> copy.(_∂ℓπ∂θ(x))
    
    neg_ℓπ = x -> -logpdf(target, x)
    _hess_func = VecTargets.gen_hess(neg_ℓπ, initial_θ) # x -> (value, gradient, hessian)
    hess_func = x -> copy.(_hess_func(x))
    
    fstabilize = H -> H + hps.λ * I
    G = x -> begin
        H = fstabilize(hess_func(x)[3])
        any(.!(isfinite.(H))) ? diagm(ones(length(x))) : H
    end
    _∂G∂θ = gen_∂H∂x(neg_ℓπ, initial_θ; f=fstabilize)
    ∂G∂θ = x -> reshape_∂H∂x(copy(_∂G∂θ(x)))

    metric = hps.metric == :dense_euclidean          ? DenseEuclideanMetric(D) :
             hps.metric == :dense_riemannian         ? DenseRiemannianMetric((D,), G, ∂G∂θ) :
             hps.metric == :dense_riemannian_softabs ? DenseRiemannianMetric((D,), G, ∂G∂θ, SoftAbsMap(hps.α)) :
             @error "Unknown metric $(hps.metric)"
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    TS = EndPointTS
    
    integrator = hps.integrator == :lf  ? Leapfrog(hps.ϵ) :
                 hps.integrator == :glf ? GeneralizedLeapfrog(hps.ϵ, hps.n) :
                 @error "Unknown integrator $(hps.integrator)"

    tc = FixedNSteps(hps.L)
    
    proposal = HMCKernel(Trajectory{TS}(integrator, tc))
    
    return (; rng, hamiltonian, proposal, initial_θ)
end

function sample_target(hps; rng=MersenneTwister(1110))    
    rng, hamiltonian, proposal, initial_θ = prepare_sample(hps; rng=rng)

    samples, stats = sample(
        rng, hamiltonian, proposal, initial_θ, hps.n_samples; progress=false, verbose=true
    )
    
    return (; target, hamiltonian, samples, stats)
end