using Random, LinearAlgebra, ReverseDiff, ForwardDiff, VecTargets, Match
using Zygote: diaghessian, hessian_reverse

gen_hess_rev(func, x::AbstractVector) = VecTargets.gen_hess(func, x)

function gen_hess_hybrid_diag(func, x::AbstractVector)
    grad = VecTargets.gen_grad(func, x)
    function hess(x::AbstractVector)
        v, g = grad(x)
        h, = diaghessian(func, x)
        return v, g, Diagonal(h)
    end
    return hess
end

function gen_hess_rev_outer(func, x::AbstractVector)
    grad = VecTargets.gen_grad(func, x)
    function hess(x::AbstractVector)
        v, g = grad(x)
        return v, g, g * g'
    end
    return hess
end

function gen_hess_rev_outer_diag(func, x::AbstractVector)
    grad = VecTargets.gen_grad(func, x)
    function hess(x::AbstractVector)
        v, g = grad(x)
        return v, g, Diagonal(g .* g)
    end
    return hess
end

# Fisher information metric
function gen_∂G∂θ_rev(Vfunc, x; f = identity)
    _Hfunc = VecTargets.gen_hess(Vfunc, ReverseDiff.track.(x))
    Hfunc = x -> f(_Hfunc(x)[3])
    # QUES What's the best output format of this function?
    return x -> ReverseDiff.jacobian(Hfunc, x) # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

# TODO Refactor this using https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers
function gen_hess_fwd(func, x::AbstractVector)
    function hess(x::AbstractVector)
        return nothing, nothing, ForwardDiff.hessian(func, x)
    end
    return hess
end

function gen_hess_zygote_diag(func, x::AbstractVector)
    function hess(x::AbstractVector)
        return nothing, nothing, Diagonal(hessian_reverse(func, x))
    end
    return hess
end

function gen_hess_fwd_diag(func, x::AbstractVector)
    function hess(x::AbstractVector)
        h, = diaghessian(func, x)
        return nothing, nothing, Diagonal(h)
    end
    return hess
end

# https://discourse.julialang.org/t/autodiff-calculate-just-the-diagonal-of-the-hessian/22893/3?u=xukai92
function diaghessian_2(f::Function, x::AbstractVector{T}) where {T<:Real}
    h = similar(x)
    for i in eachindex(x)
        fxᵢ = xᵢ -> f(vcat(x[1:(i-1)], xᵢ, x[(i+1):end])) # this maybe can be optimized
        gxᵢ = xᵢ -> ForwardDiff.derivative(fxᵢ, xᵢ)
        h[i] = ForwardDiff.derivative(gxᵢ, x[i])
    end
    return h
end

function gen_hess_fwd_diag_2(func, x::AbstractVector)
    function hess(x::AbstractVector)
        h = diaghessian_2(func, x)
        return nothing, nothing, Diagonal(h)
    end
    return hess
end

function gen_hess_fwd_outer(func, x::AbstractVector)
    function hess(x::AbstractVector)
        g = ForwardDiff.gradient(func, x)
        return nothing, nothing, g * g'
    end
    return hess
end

function gen_hess_fwd_outer_diag(func, x::AbstractVector)
    function hess(x::AbstractVector)
        g = ForwardDiff.gradient(func, x)
        return nothing, nothing, Diagonal(g .* g)
    end
    return hess
end

function gen_∂G∂θ_fwd(Vfunc, x; f = identity, gen_hess = gen_hess_fwd)
    _Hfunc = gen_hess(Vfunc, x)
    Hfunc = x -> f(_Hfunc(x)[3])
    # QUES What's the best output format of this function?
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    d = length(x)
    out = zeros(eltype(x), d^2, d)
    # default output shape [∂H∂x₁; ∂H∂x₂; ...]
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
end
# 1.764 ms 
# fwd -> 5.338 μs 
# cfg -> 3.651 μs

function gen_∂G∂θ_fwd_diag(Vfunc, x; f = identity, gen_hess = gen_hess_fwd)
    _Hfunc = gen_hess(Vfunc, x)
    Hfunc = x -> f(_Hfunc(x)[3]).diag
    # QUES What's the best output format of this function?
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    T = eltype(x)
    d = length(x)
    out = zeros(T, d, d)
    # TODO check it needs a transpose
    # default output shape w/ d = 3
    # d_x1_x1_x1, d_x1_x1_x2, d_x1_x1_x3
    # d_x2_x2_x1, d_x2_x2_x2, d_x2_x2_x3
    # d_x3_x3_x1, d_x3_x3_x2, d_x3_x3_x3
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[(i-1)*d+1:i*d, :] for i = 1:d)...; dims = 3)
end

_unit_matrix(H::Diagonal) = diagm(ones(size(H, 1)))
_unit_matrix(H::AbstractMatrix) = Diagonal(ones(size(H, 1)))

function prepare_sample_target(hps, θ₀, ℓπ; hessian_approx=:exact)
    Vfunc = x -> -ℓπ(x) # potential energy is the negative log-probability
    _Hfunc = @match hessian_approx begin # x -> (value, gradient, hessian)
        :exact      => gen_hess_rev(Vfunc, θ₀)
        :diag       => gen_hess_hybrid_diag(Vfunc, θ₀)
        :outer      => gen_hess_rev_outer(Vfunc, θ₀)
        :outer_diag => gen_hess_rev_outer_diag(Vfunc, θ₀)
    end
    Hfunc = x -> copy.(_Hfunc(x)) # _Hfunc do in-place computation, copy to avoid bug

    fstabilize = H -> H + hps.λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        any(.!(isfinite.(H))) ? _unit_matrix(H) : H
    end
    gen_hess = @match hessian_approx begin
        :exact      => gen_hess_fwd
        :diag       => gen_hess_fwd_diag
        # :diag       => gen_hess_zygote_diag
        :outer      => gen_hess_fwd_outer
        :outer_diag => gen_hess_fwd_outer_diag
    end
    
    _∂G∂θfunc = @match hessian_approx begin
        :exact      => gen_∂G∂θ_fwd(Vfunc, θ₀; f = fstabilize, gen_hess = gen_hess) # size==(4, 2)
        :diag       => gen_∂G∂θ_fwd_diag(Vfunc, θ₀; f = fstabilize, gen_hess = gen_hess) # size==(2, 2)
        :outer      => gen_∂G∂θ_fwd(Vfunc, θ₀; f = fstabilize, gen_hess = gen_hess) # size==(4, 2)
        :outer_diag => gen_∂G∂θ_fwd_diag(Vfunc, θ₀; f = fstabilize, gen_hess = gen_hess) # size==(2, 2)
    end
    ∂G∂θfunc = @match hessian_approx begin
        :exact      => x -> reshape_∂G∂θ(_∂G∂θfunc(x)) # size==(2, 2, 2)
        :diag       => _∂G∂θfunc # size==(2, 2)
        :outer      => x -> reshape_∂G∂θ(_∂G∂θfunc(x)) # size==(2, 2, 2)
        :outer_diag => _∂G∂θfunc # size==(2, 2)
    end

    return Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end

function prepare_sample(hps; rng = MersenneTwister(1110))
    target = @match hps.target begin
        :gaussian  => HighDimGaussian(2)
        :banana    => Banana()
        :funnel    => Funnel()
        :funnel3   => Funnel(3)
        :funnel5   => Funnel(5)
        :funnel11  => Funnel(11)
        :funnel21  => Funnel(21)
        :funnel31  => Funnel(31)
        :funnel41  => Funnel(41)
        :funnel51  => Funnel(51)
        :funnel101 => Funnel(101)
        :funnel151 => Funnel(151)
        :funnel201 => Funnel(201)
        :funnel251 => Funnel(251)
        :funnel301 => Funnel(301)
        :spiral    => Spiral(8, 0.1)
        :mogs      => TwoDimGaussianMixtures()
        :blr       => LogisticRegression(0.01; num_obs=200, num_latent=100) # 100
        :blr24     => LogisticRegression(0.01; num_obs=200, num_latent=24)  # no 2-way interaction
        :loggcpp   => LogGaussianCoxPointProcess(8) # 64
    end

    # θ₀ = rand(rng, dim(target))
    θ₀ = rand(rng, dim(target)) * 2 .- 1 # Michael's version, q_i ~ U(-1, 1)

    ℓπ = VecTargets.gen_logpdf(target)
    ∂ℓπ∂θ = VecTargets.gen_logpdf_grad(target, θ₀)

    _, _, Gfunc, ∂G∂θfunc = prepare_sample_target(hps, θ₀, ℓπ; hessian_approx = (:hessian_approx in keys(hps) ? hps.hessian_approx : :exact))

    D = dim(target)
    metric =
        hps.metric == :dense_euclidean ? DenseEuclideanMetric(D) :
        hps.metric == :dense_riemannian ? DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc) :
        hps.metric == :dense_riemannian_softabs ?
        DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, SoftAbsMap(hps.α)) :
        @error "Unknown metric $(hps.metric)"
    
    kinetic = get(hps, :kinetic, :gaussian) == :gaussian ? GaussianKinetic() : # support missing kinetic in hps
        hps.kinetic == :relativistic ? RelativisticKinetic(hps.m, hps.c) :
        hps.kinetic == :dimwise_relativistic ? DimensionwiseRelativisticKinetic(hps.m, hps.c) :
        @error "Unknown kinetic $(hps.kinetic)"

    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    TS = if :trajectory_sampling in keys(hps)
        @match hps.trajectory_sampling begin
            :mh => EndPointTS
            :multinomial => MultinomialTS
        end
    else
        EndPointTS
    end

    integrator =
        hps.integrator == :lf ? Leapfrog(hps.ϵ) :
        hps.integrator == :glf ? GeneralizedLeapfrog(hps.ϵ, hps.n) :
        @error "Unknown integrator $(hps.integrator)"

    tc = FixedNSteps(hps.L)

    proposal = HMCKernel(Trajectory{TS}(integrator, tc))

    return (; rng, target, hamiltonian, proposal, θ₀)
end

function sample_target(hps; rng = MersenneTwister(1110), output_only::Bool=false)
    rng, target, hamiltonian, proposal, θ₀ = prepare_sample(hps; rng = rng)

    samples, stats = sample(
        rng,
        hamiltonian,
        proposal,
        θ₀,
        hps.n_samples;
        progress = true,
        verbose = true,
    )
    if output_only
        return (; samples, stats)
    else
        return (; target, hamiltonian, samples, stats)
    end
end

function simulate_trajectory(hps; rng = MersenneTwister(1110), initial_momentum = :default)
    rng, target, hamiltonian, proposal, θ₀ = prepare_sample(hps; rng = rng)
    
    z₀ = initial_momentum isa AbstractVector ? phasepoint(hamiltonian, θ₀, initial_momentum) :
        initial_momentum == :default ? phasepoint(rng, θ₀, hamiltonian) :
        # NOTE The two options below ensures intiial momentum is identical
        initial_momentum == :simple ? phasepoint(hamiltonian, θ₀, randn(rng, size(θ₀)...)) :
        initial_momentum == :gaussian_kinetic ? phasepoint(hamiltonian, θ₀, rand(rng, hamiltonian.metric, GaussianKinetic(), θ₀)) :
        error("Unkown initial_momentum = $initial_momentum")
    z_lst = step(
        proposal.τ.integrator,
        hamiltonian,
        z₀,
        hps.L;
        full_trajectory = Val(true),
    )
    pushfirst!(z_lst, z₀)

    return (; target, hamiltonian, z_lst)
end
