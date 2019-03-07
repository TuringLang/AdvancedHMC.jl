function find_good_eps(rng::AbstractRNG, h::Hamiltonian, θ::AbstractVector{T}; max_n_iters::Int=10) where {T<:Real}
    ϵ = 0.1
    r = rand_momentum(rng, h)

    _H = hamiltonian_energy(h, θ, r)

    θ′, r′, _is_valid = step(Leapfrog(ϵ), h, θ, r)
    _H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

    ΔH = _H - _H_new
    direction = ΔH > log(0.8) ? 1 : -1

    i = 1
    # Heuristically find optimal ϵ
    while (i <= max_n_iters)
        θ = θ′

        r = rand_momentum(rng, h)
        _H = hamiltonian_energy(h, θ, r)

        θ′, r′, _is_valid = step(Leapfrog(ϵ), h, θ, r)
        _H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

        ΔH = _H - _H_new

        if ((direction == 1) && !(ΔH > log(0.8)))
            break
        elseif ((direction == -1) && !(ΔH < log(0.8)))
            break
        else
            ϵ = direction == 1 ? 2.0 * ϵ : 0.5 * ϵ
        end

        i += 1
    end

    while _H_new == Inf     # revert if the last change is too big
        ϵ = ϵ / 2           # safe is more important than large
        θ′, r′, _is_valid = step(Leapfrog(ϵ), h, θ, r)
        _H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf
    end

    return ϵ
end

find_good_eps(h::Hamiltonian, θ::AbstractVector{T}; max_n_iters::Int=10) where {T<:Real} = find_good_eps(GLOBAL_RNG, h, θ; max_n_iters=max_n_iters)
