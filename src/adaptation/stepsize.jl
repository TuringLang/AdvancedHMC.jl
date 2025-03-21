### Mutable states

mutable struct DAState{T<:AbstractScalarOrVec{<:AbstractFloat}}
    m::Int
    ϵ::T
    μ::T
    x_bar::T
    H_bar::T
end

computeμ(ϵ::AbstractScalarOrVec{<:AbstractFloat}) = log.(10 * ϵ)

function DAState(ϵ::T) where {T}
    μ = computeμ(ϵ)
    return DAState(0, ϵ, μ, zero(T), zero(T))
end

function DAState(ϵ::AbstractVector{T}) where {T}
    n = length(ϵ)
    μ = computeμ(ϵ)
    return DAState(0, ϵ, μ, zeros(T, n), zeros(T, n))
end

function reset!(das::DAState{T}) where {T<:AbstractFloat}
    das.m = 0
    das.μ = computeμ(das.ϵ)
    das.x_bar = zero(T)
    return das.H_bar = zero(T)
end

function reset!(das::DAState{<:AbstractVector{T}}) where {T<:AbstractFloat}
    das.m = 0
    das.μ .= computeμ(das.ϵ)
    das.x_bar .= zero(T)
    return das.H_bar .= zero(T)
end

mutable struct MSSState{T<:AbstractScalarOrVec{<:AbstractFloat}}
    ϵ::T
end

### Step size adaptors

abstract type StepSizeAdaptor <: AbstractAdaptor end

initialize!(adaptor::StepSizeAdaptor, n_adapts::Int) = nothing
finalize!(adaptor::StepSizeAdaptor) = nothing

getϵ(ss::StepSizeAdaptor) = ss.state.ϵ

struct FixedStepSize{T<:AbstractScalarOrVec{<:AbstractFloat}} <: StepSizeAdaptor
    ϵ::T
end
Base.show(io::IO, a::FixedStepSize) = print(io, "FixedStepSize($(a.ϵ))")

getϵ(fss::FixedStepSize) = fss.ϵ

struct ManualSSAdaptor{T<:AbstractScalarOrVec{<:AbstractFloat}} <: StepSizeAdaptor
    state::MSSState{T}
end
Base.show(io::IO, a::ManualSSAdaptor) = print(io, "ManualSSAdaptor()")

function ManualSSAdaptor(initϵ::T) where {T<:AbstractScalarOrVec{<:AbstractFloat}}
    return ManualSSAdaptor{T}(MSSState(initϵ))
end

"""
An implementation of the Nesterov dual averaging algorithm to tune step size.

References

Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623.
Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems. Mathematical programming, 120(1), 221-259.
"""
struct NesterovDualAveraging{T<:AbstractFloat,S<:AbstractScalarOrVec{T}} <: StepSizeAdaptor
    γ::T
    t_0::T
    κ::T
    δ::T
    state::DAState{S}
end
function Base.show(io::IO, a::NesterovDualAveraging)
    return print(
        io,
        "NesterovDualAveraging(γ=$(a.γ), t_0=$(a.t_0), κ=$(a.κ), δ=$(a.δ), state.ϵ=$(getϵ(a)))",
    )
end

function NesterovDualAveraging(
    γ::T, t_0::T, κ::T, δ::T, ϵ::VT
) where {T<:AbstractFloat,VT<:AbstractScalarOrVec{T}}
    return NesterovDualAveraging(γ, t_0, κ, δ, DAState(ϵ))
end

function NesterovDualAveraging(
    δ::T, ϵ::VT
) where {T<:AbstractFloat,VT<:AbstractScalarOrVec{T}}
    return NesterovDualAveraging(T(0.05), T(10.0), T(0.75), δ, ϵ)
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
# Note: This function is not merged with `adapt!` to empahsize the fact that
#       step size adaptation is not dependent on `θ`.
function adapt_stepsize!(
    da::NesterovDualAveraging{T}, α::AbstractScalarOrVec{<:T}
) where {T<:AbstractFloat}
    @debug "Adapting step size..." α

    # Clip average MH acceptance probability
    if α isa AbstractVector
        α[α .> 1] .= one(T)
    else
        α = α > 1 ? one(T) : α
    end

    (; state, γ, t_0, κ, δ) = da
    (; μ, m, x_bar, H_bar) = state

    m = m + 1

    η_H = one(T) / (m + t_0)
    H_bar = (one(T) - η_H) * H_bar .+ η_H * (δ .- α)

    x = μ .- H_bar * sqrt(m) / γ     # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (one(T) - η_x) * x_bar .+ η_x * x

    ϵ = exp.(x)
    @debug "Adapting step size..." new_ϵ = ϵ old_ϵ = da.state.ϵ

    # TODO: we might want to remove this when all other numerical issues are correctly handelled
    if !all(isfinite, ϵ)
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(da.state.ϵ) is used instead."
        # FIXME: this revert is buggy for batch mode
        (; m, ϵ, x_bar, H_bar) = state
    end

    state.m = m
    state.ϵ = ϵ
    state.x_bar = x_bar
    state.H_bar = H_bar
    return nothing
end

function adapt!(
    da::NesterovDualAveraging,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
)
    adapt_stepsize!(da, α)
    return nothing
end

reset!(da::NesterovDualAveraging) = reset!(da.state)

function finalize!(da::NesterovDualAveraging)
    da.state.ϵ = exp.(da.state.x_bar)
    return nothing
end
