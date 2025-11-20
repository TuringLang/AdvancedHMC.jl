using Random, LinearAlgebra, ReverseDiff, ForwardDiff, MCMCLogDensityProblems

# Fisher information metric
function gen_∂G∂θ_rev(Vfunc, x; f=identity)
    Hfunc = gen_hess_fwd(Vfunc, ReverseDiff.track.(x))
    
    # QUES What's the best output format of this function?
    return x -> ReverseDiff.jacobian(x -> f(Hfunc(x)), x) # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

# TODO Refactor this using https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers
function gen_hess_fwd_precompute_cfg(func, x::AbstractVector)
    cfg = ForwardDiff.HessianConfig(func, x)
    H = Matrix{eltype(x)}(undef, length(x), length(x))

    function hess(x::AbstractVector)
        ForwardDiff.hessian!(H, func, x, cfg)
        return H
    end
    return hess
end

function gen_hess_fwd(func, x::AbstractVector)
    cfg = nothing
    H = nothing
    
    function hess(x::AbstractVector)
        if cfg === nothing
            cfg = ForwardDiff.HessianConfig(func, x)
            H = Matrix{eltype(x)}(undef, length(x), length(x))
        end
        ForwardDiff.hessian!(H, func, x, cfg)
        return H
    end
    return hess
end

function gen_∂G∂θ_fwd(Vfunc, x; f=identity)
    Hfunc = gen_hess_fwd(Vfunc, x)

    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    d = length(x)
    out = zeros(eltype(x), d^2, d)

    function ∂G∂θ_fwd(y)
        ForwardDiff.jacobian!(out, Hfunc, y, cfg)
        return out
    end
    return ∂G∂θ_fwd
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return reshape(H, d, d, :)
end

function prepare_sample_target(hps, θ₀, ℓπ)
    Vfunc = x -> -ℓπ(x) # potential energy is the negative log-probability
    Hfunc = gen_hess_fwd_precompute_cfg(Vfunc, θ₀) # x -> (value, gradient, hessian)

    fstabilize = H -> begin
        @inbounds for i in 1:size(H,1)
            H[i,i] += hps.λ
        end
        H
    end
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x))
        all(isfinite, H) ? H : diagm(ones(length(x)))
    end
    _∂G∂θfunc = gen_∂G∂θ_fwd(Vfunc, θ₀; f=fstabilize) # size==(4, 2)
    ∂G∂θfunc = x -> reshape_∂G∂θ(_∂G∂θfunc(x)) # size==(2, 2, 2)

    return Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end

function prepare_sample(hps; rng=MersenneTwister(1110))
    target = if hps.target == :gaussian
        HighDimGaussian(2)
    elseif hps.target == :banana
        Banana()
    elseif hps.target == :funnel
        Funnel()
    elseif hps.target == :funnel101
        Funnel(101)
    elseif hps.target == :spiral
        Spiral(8, 0.1)
    elseif hps.target == :mogs
        TwoDimGaussianMixtures()
    else
        @error "Unknown target $(hps.target)"
    end

    θ₀ = rand(rng, dim(target))

    ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
    ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, θ₀)

    _, _, Gfunc, ∂G∂θfunc = prepare_sample_target(hps, θ₀, ℓπ)

    D = dim(target)
    metric = if hps.metric == :dense_euclidean
        DenseEuclideanMetric(D)
    elseif hps.metric == :dense_riemannian
        DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc)
    elseif hps.metric == :dense_riemannian_softabs
        DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, SoftAbsMap(hps.α))
    else
        @error "Unknown metric $(hps.metric)"
    end
    kinetic = GaussianKinetic()

    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    TS = EndPointTS

    integrator = if hps.integrator == :lf
        Leapfrog(hps.ϵ)
    elseif hps.integrator == :glf
        GeneralizedLeapfrog(hps.ϵ, hps.n)
    else
        @error "Unknown integrator $(hps.integrator)"
    end

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
