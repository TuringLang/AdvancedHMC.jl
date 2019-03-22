function find_good_eps(rng::AbstractRNG, h::Hamiltonian, θ::AbstractVector{T}; max_n_iters::Int=100) where {T<:Real}
    ϵ′ = ϵ = 0.1
    a_min, a_cross, a_max = 0.25, 0.5, 0.75 # minimal, crossing, maximal accept ratio
    d = 2.0

    r = rand_momentum(rng, h)
    H = hamiltonian_energy(h, θ, r)

    θ′, r′, _is_valid = step(Leapfrog(ϵ), h, θ, r)
    H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

    ΔH = H - H_new
    direction = ΔH > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_n_iters
        ϵ′ = direction == 1 ? d * ϵ : 1 / d * ϵ
        θ′, r′, _is_valid = step(Leapfrog(ϵ′), h, θ′, r′)
        H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

        ΔH = H - H_new
        DEBUG && @debug "Crossing step" direction H_new ϵ "α = $(min(1, exp(ΔH)))"
        if (direction == 1) && !(ΔH > log(a_cross))
            break
        elseif (direction == -1) && !(ΔH < log(a_cross))
            break
        else
            ϵ = ϵ′
        end
    end

    # Bisection step: ensure final accept ratio: a_min < a < a_max.
    # See https://en.wikipedia.org/wiki/Bisection_method
    ϵ, ϵ′ = ϵ < ϵ′ ? (ϵ, ϵ′) : (ϵ′, ϵ)  # ensure ϵ < ϵ′
    for _ = 1:max_n_iters
        ϵ_mid = middle(ϵ, ϵ′)
        θ′, r′, _is_valid = step(Leapfrog(ϵ_mid), h, θ, r)
        H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

        ΔH = H - H_new
        DEBUG && @debug "Bisection step" H_new ϵ_mid "α = $(min(1, exp(ΔH)))"
        if (exp(ΔH) > a_max)
            ϵ = ϵ_mid
        elseif (exp(ΔH) < a_min)
            ϵ′ = ϵ_mid
        else
            ϵ = ϵ_mid
            break
        end
    end

    return ϵ
end

find_good_eps(h::Hamiltonian, θ::AbstractVector{T}; max_n_iters::Int=100) where {T<:Real} = find_good_eps(GLOBAL_RNG, h, θ; max_n_iters=max_n_iters)

######################
### Mutable states ###
######################

mutable struct DAState{TI<:Integer,TF<:Real}
    m     :: TI
    ϵ     :: TF
    μ     :: TF
    x_bar :: TF
    H_bar :: TF
end

function DAState(ϵ::Real)
    μ = computeμ(ϵ)
    return DAState(0, ϵ, μ, 0.0, 0.0)
end

function computeμ(ϵ::Real)
    return log(10 * ϵ) # see NUTS paper sec 3.2.1
end

function reset!(dastate::DAState{TI,TF}) where {TI<:Integer,TF<:Real}
    dastate.μ = computeμ(da.state.ϵ)
    dastate.m = zero(TI)
    dastate.x_bar = zero(TF)
    dastate.H_bar = zero(TF)
end

mutable struct MSSState{T<:Real}
    ϵ :: T
end

################
### Adapters ###
################

abstract type StepSizeAdapter <: AbstractAdapter end

struct FixedStepSize{T<:Real} <: StepSizeAdapter
    ϵ :: T
end

function getss(fss::FixedStepSize)
    return fss.ϵ
end

struct DualAveraging{TI<:Integer,TF<:Real} <: StepSizeAdapter
  γ     :: TF
  t_0   :: TF
  κ     :: TF
  δ     :: TF
  state :: DAState{TI,TF}
end

function DualAveraging(γ::AbstractFloat, t_0::AbstractFloat, κ::AbstractFloat, δ::AbstractFloat, ϵ::AbstractFloat)
    return DualAveraging(γ, t_0, κ, δ, DAState(ϵ))
end

function DualAveraging(δ::AbstractFloat, ϵ::AbstractFloat)
    return DualAveraging(0.05, 10.0, 0.75, δ, ϵ)
end

function getss(da::DualAveraging)
    return da.state.ϵ
end

struct ManualSSAdapter{T<:Real} <:StepSizeAdapter
    state :: MSSState{T}
end

function getss(mssa::ManualSSAdapter)
    return mssa.state.ϵ
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
function adapt_stepsize!(da::DualAveraging, α::AbstractFloat)
    DEBUG && @debug "Adapting step size..." α
    da.state.m += 1
    m = da.state.m

    # Clip average MH acceptance probability
    α = α > 1 ? 1 : α

    γ = da.γ; t_0 = da.t_0; κ = da.κ; δ = da.δ
    μ = da.state.μ; x_bar = da.state.x_bar; H_bar = da.state.H_bar

    η_H = 1.0 / (m + t_0)
    H_bar = (1.0 - η_H) * H_bar + η_H * (δ - α)

    x = μ - H_bar * sqrt(m) / γ     # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (1.0 - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    DEBUG && @debug "Adapting step size..." "new ϵ = $ϵ" "old ϵ = $(da.state.ϵ)"

    if isnan(ϵ) || isinf(ϵ)
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(da.state.ϵ) is used instead."
        ϵ = da.state.ϵ
        x_bar = da.state.x_bar
        H_bar = da.state.H_bar
    end

    da.state.ϵ = ϵ
    da.state.x_bar = x_bar
    da.state.H_bar = H_bar
end

function adapt!(da::DualAveraging, θ::AbstractVector{<:Real}, α::AbstractFloat)
    adapt_stepsize!(da, α)
end

function update(h::Hamiltonian, prop::AbstractProposal, da::DualAveraging)
    return h, prop(getss(da))
end
