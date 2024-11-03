using QuasiMonteCarlo, Distributions

mutable struct Quasi_MC_seed
    array::AbstractVecOrMat{<:Real}
    counter::Int

    # Custom constructor with a default value for `counter`
    function Quasi_MC_seed(array::AbstractVecOrMat{<:Real}, counter::Int = 1)
        new(array, counter)
    end
end

function get_next_vector(q::Quasi_MC_seed; normalized = true)
    val = q.array[:, q.counter]
    q.counter += 1
    if normalized
        return uniform_to_normal(val)
    else
        return val
    end
end

function uniform_to_normal(uniform_vector::Vector{Float64})::Vector{Float64}
    normal_dist = Normal(0, 1)
    return [quantile(normal_dist, u) for u in uniform_vector]
end