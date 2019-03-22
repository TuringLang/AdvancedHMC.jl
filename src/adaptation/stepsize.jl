######################
### Mutable states ###
######################

mutable struct DAState{T<:AbstractFloat}
    m     :: Int
    ϵ     :: T
    μ     :: T
    x_bar :: T
    H_bar :: T
end

function DAState(ϵ::AbstractFloat)
    μ = computeμ(ϵ)
    return DAState(0, ϵ, μ, 0.0, 0.0)
end

function computeμ(ϵ::AbstractFloat)
    return log(10 * ϵ) # see NUTS paper sec 3.2.1
end

function reset!(dastate::DAState{T}) where {T<:AbstractFloat}
    dastate.μ = computeμ(da.state.ϵ)
    dastate.m = 0
    dastate.x_bar = zero(T)
    dastate.H_bar = zero(T)
end

mutable struct MSSState{T<:AbstractFloat}
    ϵ :: T
end

################
### Adapters ###
################

abstract type StepSizeAdapter <: AbstractAdapter end

struct FixedStepSize{T<:AbstractFloat} <: StepSizeAdapter
    ϵ :: T
end

function getss(fss::FixedStepSize)
    return fss.ϵ
end

struct DualAveraging{T<:AbstractFloat} <: StepSizeAdapter
  γ     :: T
  t_0   :: T
  κ     :: T
  δ     :: T
  state :: DAState{T}
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

struct ManualSSAdapter{T<:AbstractFloat} <:StepSizeAdapter
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

function adapt!(da::DualAveraging, θ::AbstractVector{<:AbstractFloat}, α::AbstractFloat)
    adapt_stepsize!(da, α)
end
