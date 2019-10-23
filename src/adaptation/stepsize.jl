######################
### Mutable states ###
######################

mutable struct DAState{T<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}}
    m     :: Int
    ϵ     :: T
    μ     :: T
    x_bar :: T
    H_bar :: T
end

computeμ(ϵ::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}) = log.(10 * ϵ)

function DAState(ϵ::AbstractFloat)
    μ = computeμ(ϵ)
    return DAState(0, ϵ, μ, 0.0, 0.0)
end

function DAState(ϵ::AbstractVector{<:AbstractFloat})
    n = length(ϵ)
    μ = computeμ(ϵ)
    return DAState(0, ϵ, μ, zeros(n), zeros(n))
end

function reset!(dastate::DAState{T}) where {T<:AbstractFloat}
    dastate.m = 0
    dastate.μ = computeμ(dastate.ϵ)
    dastate.x_bar = zero(T)
    dastate.H_bar = zero(T)
end

function reset!(dastate::DAState{VT}) where {VT<:AbstractVector{<:AbstractFloat}}
    T = eltype(VT)
    dastate.m = 0
    dastate.μ .= computeμ(dastate.ϵ)
    dastate.x_bar .= zero(T)
    dastate.H_bar .= zero(T)
end

mutable struct MSSState{T<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}}
    ϵ :: T
end

################
### Adaptors ###
################

abstract type StepSizeAdaptor <: AbstractAdaptor end

# finalize!(adaptor::T) where {T<:StepSizeAdaptor} = nothing

getϵ(ss::StepSizeAdaptor) = ss.state.ϵ

struct FixedStepSize{T<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}} <: StepSizeAdaptor
    ϵ :: T
end
Base.show(io::IO, a::FixedStepSize) = print(io, "FixedStepSize($(a.ϵ))")

getϵ(fss::FixedStepSize) = fss.ϵ

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
  state :: DAState{<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}}
end
Base.show(io::IO, a::NesterovDualAveraging) = print(io, "NesterovDualAveraging(γ=$(a.γ), t_0=$(a.t_0), κ=$(a.κ), δ=$(a.δ), state.ϵ=$(getϵ(a)))")

NesterovDualAveraging(
    γ::T,
    t_0::T,
    κ::T,
    δ::T,
    ϵ::VT
) where {T<:AbstractFloat, VT<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}} = NesterovDualAveraging(γ, t_0, κ, δ, DAState(ϵ))
NesterovDualAveraging(
    δ::T,
    ϵ::VT
) where {T<:AbstractFloat, VT<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}} = NesterovDualAveraging(0.05, 10.0, 0.75, δ, ϵ)

struct ManualSSAdaptor{T<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}} <:StepSizeAdaptor
    state :: MSSState{T}
end
Base.show(io::IO, a::ManualSSAdaptor) = print(io, "ManualSSAdaptor()")

ManualSSAdaptor(initϵ::T) where {T<:Union{AbstractFloat,AbstractVector{<:AbstractFloat}}} = ManualSSAdaptor{T}(MSSState(initϵ))

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
# Note: This function is not merged with `adapt!` to empahsize the fact that
#       step size adaptation is not dependent on `θ`.
function adapt_stepsize!(da::NesterovDualAveraging{T}, α::T) where {T<:AbstractFloat}
    DEBUG && @debug "Adapting step size..." α

    # Clip average MH acceptance probability
    α = α > 1 ? one(T) : α

    @unpack state, γ, t_0, κ, δ = da
    @unpack μ, m, x_bar, H_bar = state

    m = m + 1

    η_H = one(T) / (m + t_0)
    H_bar = (one(T) - η_H) * H_bar + η_H * (δ - α)

    x = μ - H_bar * sqrt(m) / γ     # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (one(T) - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    DEBUG && @debug "Adapting step size..." "new ϵ = $ϵ" "old ϵ = $(da.state.ϵ)"

    # TODO: we might want to remove this when all other numerical issues are correctly handelled
    if isnan(ϵ) || isinf(ϵ)
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(da.state.ϵ) is used instead."
        @unpack m, ϵ, x_bar, H_bar = state
    end

    @pack! state = m, ϵ, x_bar, H_bar
end

# TODO: merge two `adapt_stepsize!` if broadcast doesn't have overhead
function adapt_stepsize!(da::NesterovDualAveraging{T}, α::VT) where {T,VT<:AbstractVector{<:AbstractFloat}}
    DEBUG && @debug "Adapting step size..." α
    @assert T == eltype(VT) "Types of `NesterovDualAveraging` and `α` are not matached."

    # Clip average MH acceptance probability
    idx = α .> 1
    α[idx] .= one(T)

    @unpack state, γ, t_0, κ, δ = da
    @unpack μ, m, x_bar, H_bar = state

    m = m + 1

    η_H = one(T) / (m + t_0)
    H_bar = (one(T) - η_H) * H_bar .+ η_H * (δ .- α)

    x = μ .- H_bar * sqrt(m) / γ     # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (one(T) - η_x) * x_bar .+ η_x * x

    ϵ = exp.(x)
    DEBUG && @debug "Adapting step size..." "new ϵ = $ϵ" "old ϵ = $(da.state.ϵ)"

    # TODO: we might want to remove this when all other numerical issues are correctly handelled
    if any(isnan.(ϵ)) || any(isinf.(ϵ))
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(da.state.ϵ) is used instead."
        # TODO: this revert is buggy for batch mode
        @unpack m, ϵ, x_bar, H_bar = state
    end

    @pack! state = m, ϵ, x_bar, H_bar
end

adapt!(
    da::NesterovDualAveraging,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}
) = adapt_stepsize!(da, α)

reset!(da::NesterovDualAveraging) = reset!(da.state)

function finalize!(da::NesterovDualAveraging)
    da.state.ϵ = exp.(da.state.x_bar)
end
