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
    dastate.μ = computeμ(dastate.ϵ)
    dastate.m = 0
    dastate.x_bar = zero(T)
    dastate.H_bar = zero(T)
end

mutable struct MSSState{T<:AbstractFloat}
    ϵ :: T
end

################
### Adaptors ###
################

abstract type StepSizeAdaptor <: AbstractAdaptor end

struct FixedStepSize{T<:AbstractFloat} <: StepSizeAdaptor
    ϵ :: T
end

function getϵ(fss::FixedStepSize)
    return fss.ϵ
end

"""
An implementation of the Nesterov dual averaging algorithm to tune step size.

References

Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623.
Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems. Mathematical programming, 120(1), 221-259.
"""
struct NesterovDualAveraging{T<:AbstractFloat} <: StepSizeAdaptor
  γ     :: T
  t_0   :: T
  κ     :: T
  δ     :: T
  state :: DAState{T}
end

reset!(da::NesterovDualAveraging) = reset!(da.state)

function NesterovDualAveraging(γ::AbstractFloat, t_0::AbstractFloat, κ::AbstractFloat, δ::AbstractFloat, ϵ::AbstractFloat)
    return NesterovDualAveraging(γ, t_0, κ, δ, DAState(ϵ))
end

function NesterovDualAveraging(δ::AbstractFloat, ϵ::AbstractFloat)
    return NesterovDualAveraging(0.05, 10.0, 0.75, δ, ϵ)
end

function getϵ(da::NesterovDualAveraging)
    return da.state.ϵ
end

struct ManualSSAdaptor{T<:AbstractFloat} <:StepSizeAdaptor
    state :: MSSState{T}
end

function getϵ(mssa::ManualSSAdaptor)
    return mssa.state.ϵ
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
# TODO: merge this function with adapt!
function adapt_stepsize!(da::NesterovDualAveraging, α::AbstractFloat)
    DEBUG && @debug "Adapting step size..." α
    
    # Clip average MH acceptance probability
    α = α > 1 ? 1 : α

    m = da.state.m; γ = da.γ; t_0 = da.t_0; κ = da.κ; δ = da.δ
    μ = da.state.μ; x_bar = da.state.x_bar; H_bar = da.state.H_bar

    m = m + 1

    η_H = 1.0 / (m + t_0)
    H_bar = (1.0 - η_H) * H_bar + η_H * (δ - α)

    x = μ - H_bar * sqrt(m) / γ     # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (1.0 - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    DEBUG && @debug "Adapting step size..." "new ϵ = $ϵ" "old ϵ = $(da.state.ϵ)"

    # TODO: we might want to remove this when all other numerical issues are correctly handelled
    if isnan(ϵ) || isinf(ϵ)
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(da.state.ϵ) is used instead."
        m = da.state.m
        ϵ = da.state.ϵ
        x_bar = da.state.x_bar
        H_bar = da.state.H_bar
    end

    da.state.m = m
    da.state.ϵ = ϵ
    da.state.x_bar = x_bar
    da.state.H_bar = H_bar
end

function adapt!(da::NesterovDualAveraging, θ::AbstractVector{<:AbstractFloat}, α::AbstractFloat)
    adapt_stepsize!(da, α)
end

function finalize!(da::NesterovDualAveraging)
    da.state.ϵ = exp(da.state.x_bar)
end
