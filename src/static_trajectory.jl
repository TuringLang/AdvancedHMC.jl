function transition(rng, h, τ::Trajectory{I, <:FixedNSteps}, ::Type{TS}, z) where {I, TS}
    @unpack integrator, term_criterion = τ
    H0 = energy(z)
    z′, is_accept, α = propose_phasepoint(rng, integrator, term_criterion, TS, h, z)
    # Do the actual accept / reject
    # NOTE: this function changes `z′` in-place in the vectorized mode
    z = accept_phasepoint!(z, z′, is_accept)
    # Reverse momentum variable to preserve reversibility
    z = PhasePoint(z.θ, -z.r, z.ℓπ, z.ℓκ)
    H = energy(z)
    tstat = merge(
        (
            n_steps = term_criterion.n_steps,
            is_accept = is_accept,
            acceptance_rate = α,
            log_density = z.ℓπ.value,
            hamiltonian_energy = H,
            hamiltonian_energy_error = H - H0,
        ),
        stat(integrator),
    )
    return Transition(z, tstat)
end

function transition(rng, h, τ::Trajectory{I, <:FixedLength}, ::Type{TS}, z) where {I, TS}
    @unpack integrator, term_criterion = τ
    # Create the corresponding `FixedNSteps` term_criterion
    n_steps = max(1, floor(Int, term_criterion.λ / nom_step_size(integrator)))
    τ = Trajectory(integrator, FixedNSteps(n_steps))
    return transition(rng, τ, TS, h, z)
end

"Use end-point from the trajectory as a proposal and apply MH correction"
function propose_phasepoint(rng, integrator, tc, ::Type{MetropolisTS}, h, z)
    z′ = step(integrator, h, z, tc.n_steps)
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    return z′, is_accept, α
end

"Perform MH acceptance based on energy, i.e. negative log probability"
function mh_accept_ratio(rng, Horiginal::T, Hproposal::T) where {T<:Real}
    α = min(one(T), exp(Horiginal - Hproposal))
    accept = rand(rng, T) < α
    return accept, α
end

function mh_accept_ratio(rng, Horiginal::T, Hproposal::T) where {T<:AbstractVector{<:Real}}
    α = min.(one(T), exp.(Horiginal .- Hproposal))
    # NOTE: There is a chance that sharing the RNG over multiple
    #       chains for accepting / rejecting might couple
    #       the chains. We need to revisit this more rigirously 
    #       in the future. See discussions at 
    #       https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    accept = rand(rng, T, length(Horiginal)) .< α
    return accept, α
end

"Propose a point from the trajectory using Multinomial sampling"
function propose_phasepoint(rng, integrator, tc, ::Type{MultinomialTS}, h, z)
    n_steps = abs(tc.n_steps)
    # TODO: Deal with vectorized-mode generically.
    #       Currently the direction of multiple chains are always coupled
    n_steps_fwd = rand_coupled(rng, 0:n_steps) 
    zs_fwd = step(integrator, h, z, n_steps_fwd; fwd=true,  full_trajectory=Val(true))
    n_steps_bwd = n_steps - n_steps_fwd
    zs_bwd = step(integrator, h, z, n_steps_bwd; fwd=false, full_trajectory=Val(true))
    zs = vcat(reverse(zs_bwd)..., z, zs_fwd...)
    ℓweights = -energy.(zs)
    if eltype(ℓweights) <: AbstractVector
        ℓweights = cat(ℓweights...; dims=2)
    end
    unnorm_ℓprob = ℓweights
    z′ = multinomial_sample(rng, zs, unnorm_ℓprob)
    # Computing adaptation statistics for dual averaging as done in NUTS
    Hs = -ℓweights
    ΔH = Hs .- energy(z)
    α = exp.(min.(0, -ΔH))  # this is a matrix for vectorized mode and a vector otherwise
    α = typeof(α) <: AbstractVector ? mean(α) : vec(mean(α; dims=2))
    return z′, true, α
end

"Sample `i` from a Categorical with unnormalised probability `unnorm_ℓp` and return `zs[i]`"
function multinomial_sample(rng, zs, unnorm_ℓp::AbstractVector)
    p = exp.(unnorm_ℓp .- logsumexp(unnorm_ℓp))
    i = randcat(rng, p)
    return zs[i]
end

# Note: zs is in the form of Vector{PhasePoint{Matrix}} and has shape [n_steps][dim, n_chains]
function multinomial_sample(rng, zs, unnorm_ℓP::AbstractMatrix)
    z = similar(first(zs))
    P = exp.(unnorm_ℓP .- logsumexp(unnorm_ℓP; dims=2)) # (n_chains, n_steps)
    is = randcat(rng, P')
    foreach(enumerate(is)) do (i_chain, i_step)
        zi = zs[i_step]
        z.θ[:,i_chain] = zi.θ[:,i_chain]
        z.r[:,i_chain] = zi.r[:,i_chain]
        z.ℓπ.value[i_chain] = zi.ℓπ.value[i_chain]
        z.ℓπ.gradient[:,i_chain] = zi.ℓπ.gradient[:,i_chain]
        z.ℓκ.value[i_chain] = zi.ℓκ.value[i_chain]
        z.ℓκ.gradient[:,i_chain] = zi.ℓκ.gradient[:,i_chain]
    end
    return z
end

accept_phasepoint!(z::T, z′::T, is_accept) where {T<:PhasePoint{<:AbstractVector}} = 
    is_accept ? z′ : z

function accept_phasepoint!(z::T, z′::T, is_accept) where {T<:PhasePoint{<:AbstractMatrix}}
    # Revert unaccepted proposals in `z′`
    is_reject = (!).(is_accept)
    if any(is_reject)
        z′.θ[:,is_reject] = z.θ[:,is_reject]
        z′.r[:,is_reject] = z.r[:,is_reject]
        z′.ℓπ.value[is_reject] = z.ℓπ.value[is_reject]
        z′.ℓπ.gradient[:,is_reject] = z.ℓπ.gradient[:,is_reject]
        z′.ℓκ.value[is_reject] = z.ℓκ.value[is_reject]
        z′.ℓκ.gradient[:,is_reject] = z.ℓκ.gradient[:,is_reject]
    end
    # NOTE: This in place treatment of `z′` is for memory efficient consideration.
    #       We can also copy `z′ and avoid mutating the original `z′`. But this is
    #       not efficient and immutability of `z′` is not important in this local scope.
    return z′
end
