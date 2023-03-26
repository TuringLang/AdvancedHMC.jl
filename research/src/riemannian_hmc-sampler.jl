using Random, LinearAlgebra, ReverseDiff, VecTargets

# Fisher information metric
function gen_∂G∂θ(Vfunc, x; f=identity)
    Hfunc = VecTargets.gen_hess(Vfunc, ReverseDiff.track.(x))
    # QUES What's the best output format of this function?
    return x -> ReverseDiff.jacobian(x -> f(Hfunc(x)[3]), x) # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[(i-1)*d+1:i*d,:] for i in 1:d)...; dims=3)
end

function prepare_sample_target(hps, θ₀, ℓπ)
    Vfunc = x -> -ℓπ(x) # potential energy is the negative log-probability
    _Hfunc = VecTargets.gen_hess(Vfunc, θ₀) # x -> (value, gradient, hessian)
    Hfunc = x -> copy.(_Hfunc(x)) # _Hfunc do in-place computation, copy to avoid bug
    
    fstabilize = H -> H + hps.λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        any(.!(isfinite.(H))) ? diagm(ones(length(x))) : H
    end
    _∂G∂θfunc = gen_∂G∂θ(Vfunc, θ₀; f=fstabilize) # size==(4, 2)
    ∂G∂θfunc = x -> reshape_∂G∂θ(_∂G∂θfunc(x)) # size==(2, 2, 2)

    return Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end

function prepare_sample(hps; rng=MersenneTwister(1110))
    target = hps.target == :gaussian  ? HighDimGaussian(2) :
             hps.target == :banana    ? Banana() :
             hps.target == :funnel    ? Funnel() :
             hps.target == :funnel101 ? Funnel(101) :
             hps.target == :spiral    ? Spiral(8, 0.1) :
             hps.target == :mogs      ? TwoDimGaussianMixtures() :
             @error "Unknown target $(hps.target)"

    θ₀ = rand(rng, dim(target))

    ℓπ = VecTargets.gen_logpdf(target)
    ∂ℓπ∂θ = VecTargets.gen_logpdf_grad(target, θ₀)
    
    _, _, Gfunc, ∂G∂θfunc = prepare_sample_target(hps, θ₀, ℓπ)

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
    
    return (; rng, target, hamiltonian, proposal, θ₀)
end

function sample_target(hps; rng=MersenneTwister(1110))    
    rng, target, hamiltonian, proposal, θ₀ = prepare_sample(hps; rng=rng)

    samples, stats = sample(
        rng, hamiltonian, proposal, θ₀, hps.n_samples; progress=false, verbose=true
    )
    
    return (; target, hamiltonian, samples, stats)
end