module AdvancedHMCMCMCChainsExt

using AdvancedHMC: AbstractMCMC, Transition, stat
using MCMCChains: Chains

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:Transition},
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    state,
    chain_type::Type{Chains};
    discard_initial=0,
    thinning=1,
    param_names=missing,
    bijector=identity,
    kwargs...,
)
    # Turn all the transitions into a vector-of-vectors.
    t = ts[1]
    tstat = merge((; lp=t.z.ℓπ.value), stat(t))
    tstat_names = collect(keys(tstat))
    vals = [vcat(bijector(t.z.θ), t.z.ℓπ.value, collect(values(stat(t)))) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1].z.θ))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    # Bundle everything up and return a Chains struct.
    return Chains(
        vals,
        vcat(param_names, tstat_names),
        (parameters=param_names, internals=tstat_names);
        start=discard_initial + 1,
        thin=thinning,
    )
end

end # module
