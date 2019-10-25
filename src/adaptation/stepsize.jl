######################
### Mutable states ###
######################

mutable struct DAState{T<:AbstractScalarOrVec{<:AbstractFloat}}
    m     :: Int
    ϵ     :: T
    μ     :: T
    x_bar :: T
    H_bar :: T
end

computeμ(ϵ::AbstractScalarOrVec{<:AbstractFloat}) = log.(10 * ϵ)

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

mutable struct MSSState{T<:AbstractScalarOrVec{<:AbstractFloat}}
    ϵ :: T
end

################
### Adaptors ###
################

abstract type StepSizeAdaptor <: AbstractAdaptor end

# finalize!(adaptor::T) where {T<:StepSizeAdaptor} = nothing

getϵ(ss::StepSizeAdaptor) = ss.state.ϵ

struct FixedStepSize{T<:AbstractScalarOrVec{<:AbstractFloat}} <: StepSizeAdaptor
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
  state :: DAState{<:AbstractScalarOrVec{<:AbstractFloat}}
end
Base.show(io::IO, a::NesterovDualAveraging) = print(io, "NesterovDualAveraging(γ=$(a.γ), t_0=$(a.t_0), κ=$(a.κ), δ=$(a.δ), state.ϵ=$(getϵ(a)))")

NesterovDualAveraging(
    γ::T,
    t_0::T,
    κ::T,
    δ::T,
    ϵ::VT
) where {T<:AbstractFloat, VT<:AbstractScalarOrVec{<:AbstractFloat}} = NesterovDualAveraging(γ, t_0, κ, δ, DAState(ϵ))
NesterovDualAveraging(
    δ::T,
    ϵ::VT
) where {T<:AbstractFloat, VT<:AbstractScalarOrVec{<:AbstractFloat}} = NesterovDualAveraging(0.05, 10.0, 0.75, δ, ϵ)

struct ManualSSAdaptor{T<:AbstractScalarOrVec{<:AbstractFloat}} <:StepSizeAdaptor
    state :: MSSState{T}
end
Base.show(io::IO, a::ManualSSAdaptor) = print(io, "ManualSSAdaptor()")

ManualSSAdaptor(initϵ::T) where {T<:AbstractScalarOrVec{<:AbstractFloat}} = ManualSSAdaptor{T}(MSSState(initϵ))

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
# Note: This function is not merged with `adapt!` to empahsize the fact that
#       step size adaptation is not dependent on `θ`.
function adapt_stepsize!(da::NesterovDualAveraging{T}, α::AbstractScalarOrVec{<:T}) where {T<:AbstractFloat}
    DEBUG && @debug "Adapting step size..." α

    # Clip average MH acceptance probability
    if α isa AbstractVector
        α[α .> 1] .= one(T)
    else
        α = α > 1 ? one(T) : α
    end

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
    α::AbstractScalarOrVec{<:AbstractFloat}
) = adapt_stepsize!(da, α)

reset!(da::NesterovDualAveraging) = reset!(da.state)

function finalize!(da::NesterovDualAveraging)
    da.state.ϵ = exp.(da.state.x_bar)
end
