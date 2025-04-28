using Random, LinearAlgebra, ReverseDiff, ForwardDiff, MCMCLogDensityProblems

# Fisher information metric
function gen_∂G∂θ_rev(Vfunc, x; f=identity)
    _Hfunc = MCMCLogDensityProblems.gen_hess(Vfunc, ReverseDiff.track.(x))
    Hfunc = x -> _Hfunc(x)[3]
    # QUES What's the best output format of this function?
    return x -> ReverseDiff.jacobian(x -> f(Hfunc(x)), x) # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

# TODO Refactor this using https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers
function gen_hess_fwd(func, x::AbstractVector)
    function hess(x::AbstractVector)
        return nothing, nothing, ForwardDiff.hessian(func, x)
    end
    return hess
end

function gen_∂G∂θ_fwd(Vfunc, x; f=identity)
    _Hfunc = gen_hess_fwd(Vfunc, x)
    Hfunc = x -> _Hfunc(x)[3]
    # QUES What's the best output format of this function?
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    d = length(x)
    out = zeros(eltype(x), d^2, d)
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
    return out # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end
# 1.764 ms 
# fwd -> 5.338 μs 
# cfg -> 3.651 μs

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[((i - 1) * d + 1):(i * d), :] for i in 1:d)...; dims=3)
end

function prepare_sample_target(hps, θ₀, ℓπ)
    Vfunc = x -> -ℓπ(x) # potential energy is the negative log-probability
    _Hfunc = MCMCLogDensityProblems.gen_hess(Vfunc, θ₀) # x -> (value, gradient, hessian)
    Hfunc = x -> copy.(_Hfunc(x)) # _Hfunc do in-place computation, copy to avoid bug

    fstabilize = H -> H + hps.λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
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
