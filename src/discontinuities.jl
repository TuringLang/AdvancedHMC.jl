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

### Quadratic Step Functions
"""
$(TYPEDEF)

A quadratic step function of the form `1(xᵀAx + bᵀx + c ≥ 0)`.

# Fields

$(TYPEDFIELDS)
"""
# TODO: add degeneracy check
struct QuadraticStepFunction <: AbstractStepFunction
    A::Matrix{<:AbstractFloat}
    b::Vector{<:AbstractFloat}
    c::AbstractFloat
end

function evaluate(s::QuadraticStepFunction, x::AbstractVector)
    return dot(x, s.A * x) + dot(s.b, x) + s.c ≥ 0
end

function intersect_line(s::QuadraticStepFunction, x0::AbstractVector, d::AbstractVector)
    A, b, c = s.A, s.b, s.c

    # Solve quadratic for intersection time
    a_coeff = dot(d, A * d)
    b_coeff = 2 * dot(x0, A * d) + dot(b, d)
    c_coeff = dot(x0, A * x0) + dot(b, x0) + c
    discrim = b_coeff^2 - 4 * a_coeff * c_coeff

    if discrim < 0
        return nothing
    end

    t_minus = (-b_coeff - sqrt(discrim)) / (2 * a_coeff)
    t_plus = (-b_coeff + sqrt(discrim)) / (2 * a_coeff)
    if t_minus < 0
        if t_plus < 0
            return nothing
        else
            t = t_plus
        end
    else
        t = t_minus
    end
    x = x0 + t * d

    normal = get_normal(s, x)
    jump_up = dot(normal, d) > 0

    return x, t, jump_up
end

function get_normal(s::QuadraticStepFunction, x0::AbstractVector)
    n = 2 * s.A * x0 + s.b
    return n / norm(n)
end

### Differentiable Step Function
"""
$(TYPEDEF)

A differentiable step function of the form `1(f(x) ≥ 0)` where f is differentiable.

# Fields

$(TYPEDFIELDS)
"""
# TODO: implement autodiff
struct DifferentiableStepFunction <: AbstractStepFunction
    f::Function
    df::Function
    tolerance::AbstractFloat
end

function evaluate(s::DifferentiableStepFunction, x::AbstractVector)
    return s.f(x) ≥ 0
end

# TODO: intersect line should accept the maximum step size
# TODO: since we have gradient information, we could use Newton's method
# TODO: this assumes there is only one intersection
# TODO: allow for inexact intersection as in Betancourt 2010
function intersect_line(s::DifferentiableStepFunction, x0::AbstractVector, d::AbstractVector)
    # Find the intersection using interval bisection
    intitial = s.f(x0) ≥ 0
    if s.f(x0 + d) ≥ 0 == intitial
        return nothing
    end
    t_min = 0.0
    t_max = 1.0
    t = 0.5
    x = x0 + t * d
    while t_max - t_min > s.tolerance
        if (s.f(x) ≥ 0) == intitial
            t_min = t
        else
            t_max = t
        end
        t = (t_min + t_max) / 2
        x = x0 + t * d
    end

    normal = get_normal(s, x)
    jump_up = dot(normal, d) > 0

    return x, t, jump_up
end

function get_normal(s::DifferentiableStepFunction, x0::AbstractVector)
    return s.df(x0) / norm(s.df(x0))
end
