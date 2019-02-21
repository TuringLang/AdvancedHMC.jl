# An Julia Implementation of Naive No-U-Turn Sampler described in Algorithm 2 in Hoffman et al. (2011)
# Author: Kai Xu
# Date: 05/10/2016

function naive_NUTS(θ0, ϵ, L, M)
  doc"""
    - θ0  : initial model parameter
    - ϵ   : leapfrog step size
    - L   : likelihood function
    - M   : sample number
  """

  function leapfrog(θ, r, ϵ)
    doc"""
      - θ : model parameter
      - r : momentum variable
      - ϵ : leapfrog step size
    """
    r̃ = r + (ϵ / 2) * ∇L(θ)
    θ̃ = θ + ϵ * r̃
    r̃ = r̃ + (ϵ / 2) * ∇L(θ̃)
    return θ̃, r̃
  end

  function build_tree(θ, r, u, v, j, ϵ)
    doc"""
      - θ   : model parameter
      - r   : momentum variable
      - u   : slice variable
      - v   : direction ∈ {-1, 1}
      - j   : depth
      - ϵ   : leapfrog step size
    """
    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′ = leapfrog(θ, r, v * ϵ)
      C′ = Set{Tuple}()
      if u <= exp(L(θ′) - 0.5 * dot(r′, r′))
        push!(C′, (θ′, r′))
      end
      s′ = u < exp(Δ_max + L(θ′) - 0.5 * dot(r′, r′))
      return θ′, r′, θ′, r′, C′, s′
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, C′, s′ = build_tree(θ, r, u, v, j - 1, ϵ)
      if v == -1
        θm, rm, _, _, C′′, s′′ = build_tree(θm, rm, u, v, j - 1, ϵ)
      else
        _, _, θp, rp, C′′, s′′ = build_tree(θp, rp, u, v, j - 1, ϵ)
      end
      s′ = s′ & s′′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
      C′ = union(C′, C′′)
      return θm, rm, θp, rp, C′, s′
    end
  end

  ∇L = θ -> ForwardDiff.gradient(L, θ)  # generate gradient function

  θ = Array{Array}(M + 1)
  θ[1] = θ0

  println("[naive_NUTS] start sampling with ϵ=$ϵ for $M samples")

  for m = 1:M
    print('.')
    r0 = randn(length(θ0))
    u = rand() * exp(L(θ[m]) - 0.5 * dot(r0, r0))  # Note: θ^{m-1} in the paper corresponds to
                                                    #       `θ[m]` in the code
    θm, θp, rm, rp, j, C, s = θ[m], θ[m], r0, r0, 0, Set{Tuple}(), true
    push!(C, (θ[m], r0))
    while s == true
      v_j = rand([-1, 1]) # Note: this variable actually does not depend on j;
                          #       it is set as `v_j` just to be consistent to the paper
      if v_j == -1
        θm, rm, _, _, C′, s′ = build_tree(θm, rm, u, v_j, j, ϵ)
      else
        _, _, θp, rp, C′, s′ = build_tree(θp, rp, u, v_j, j, ϵ)
      end
      if s′ == 1
        C = union(C, C′)
      end
      s = s′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
      j += 1
    end
    θm, r = rand(collect(C))  # Note: `rand(::Set)` is not supported; thus need
                              #       to convert Set to Array by `collect(::Set)`
    θ[m + 1] = θm
  end

  println()
  println("[naive_NUTS] sampling complete")

  return θ
end
