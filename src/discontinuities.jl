####
#### Utilites for handling discontinuous hamiltonian dynamics.
####

using LinearAlgebra

"""
$(TYPEDEF)

Represents a step function that can be used with `DiscontinuousLeapfrog`.
"""
abstract type AbstractStepFunction end

"""
    evaluate(s::AbstractStepFunction, x::AbstractVector)

Evaluate the step function `s` at the point `x`.
"""
function evaluate end

"""
    intersect_line(s::AbstractStepFunction, x0::AbstractVector, d::AbstractVector) -> Tuple{x::, t}

Returns the intersection point (`x`) and time (`t`) of the line `x0 + t * d` with the
discontinuity of the step function `s`, along with whether the intersection corresponds
to an increase in the step function (`jump_up`).

If there is no intersection, returns `nothing`.
"""
function intersect_line end

"""
    get_normal(c::AbstractStepFunction, x0::AbstractVector) -> normal::AbstractVector

Returns the normal vector of the step function's discontinuity at the point `x0`.
"""
function get_normal end

### Linear Step Functions
"""
$(TYPEDEF)

A linear step function of the form 1(aᵀx + b ≥ 0)`.

# Fields

$(TYPEDFIELDS)
"""
struct LinearStepFunction <: AbstractStepFunction
    a::Vector{<:AbstractFloat}
    b::AbstractFloat
end

function evaluate(s::LinearStepFunction, x::AbstractVector)
    return dot(s.a, x) + s.b ≥ 0
end

function intersect_line(s::LinearStepFunction, x0::AbstractVector, d::AbstractVector)
    a, b = s.a, s.b
    aᵀd = dot(a, d)

    # No intersection if lines are parallel
    if aᵀd ≈ 0
        return nothing
    end

    t = -(b + dot(a, x0)) / aᵀd
    x = x0 + t * d

    jump_up = aᵀd > 0

    return x, t, jump_up
end

function get_normal(s::LinearStepFunction, ::AbstractVector)
    return s.a / norm(s.a)
end

# struct DiscontinuousLogDensityModel <: LogDensityModel
#     logdensity::L
#     step_functions::Vector{<:AbstractStepFunction}
# end
