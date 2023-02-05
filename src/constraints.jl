####
#### Constraints for Hamiltonian dynamics.
####

using LinearAlgebra

"""
$(TYPEDEF)

Represents a constraint that can be used with `ConstrainedLeapfrog`.
"""
abstract type AbstractConstraint end

"""
    intersect_line(c::AbstractConstraint, x0::AbstractVector, d::AbstractVector) -> Tuple{x::, t}

Returns the intersection point (`x`) and time (`t`) of the line `x0 + t * d` with the
constraint `c`. If there is no intersection, returns `nothing`.
"""
function intersect_line end

"""
    get_normal(c::AbstractConstraint, x0::AbstractVector) -> normal::AbstractVector

Returns the normal vector of the constraint at the point `x0`.
"""
function get_normal end

### Linear Constraints
"""
$(TYPEDEF)

A linear constraint of the form `aᵀx + b ≥ 0`.

# Fields

$(TYPEDFIELDS)
"""
struct LinearConstraint <: AbstractConstraint
    a::Vector{<:AbstractFloat}
    b::AbstractFloat
end

function intersect_line(c::LinearConstraint, x0::AbstractVector, d::AbstractVector)
    a, b = c.a, c.b
    # Only reflect when leaving the feasibe region.
    if dot(a, d) >= 0
        return nothing
    end

    t = -(b + dot(a, x0)) / dot(a, d)
    x = x0 + t * d
    return x, t
end

function get_normal(c::LinearConstraint, x0::AbstractVector)
    return c.a / norm(c.a)
end
